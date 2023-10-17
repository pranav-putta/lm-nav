import torch
import numpy as np
import einops
from torch import nn
from transformers import CLIPVisionModel
from torchvision import transforms

from lmnav.models.base_model import BaseModel
from lmnav.models.eva_vit import create_eva_vit_g
from lmnav.models.blip2 import disabled_train
from lmnav.models.Qformer import BertConfig, BertLMHeadModel
from lmnav.models.perceiver import Perceiver

from lmnav.common.dist_utils import download_cached_file
from lmnav.common.utils import is_url, convert_weights_to_fp16
import contextlib

import logging
import os
from vc_models.models.vit import model_utils as vc_model_utils


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ObservationEncoder(BaseModel):
    def embed_obs(self, obs):
        """
        consumes a tensordict of observations and computes an embedding
        """
        raise NotImplementedError("embed_visual needs to be implemented")

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def hidden_size(self):
        raise NotImplementedError("hidden_size needs to be implemented")

    @property
    def vis_processor(self):
        raise NotImplementedError("vis_processor needs to be implemented")

    @property
    def num_tokens(self):
        raise NotImplementedError("num_tokens needs to be implemented")


class QformerObservationEncoder(ObservationEncoder):
    def __init__(
        self,
        image_size,
        vis_processor,
        vit_precision,
        vit_model,
        drop_path_rate,
        use_grad_checkpoint,
        num_query_token,
        freeze_backbone,
        freeze_qformer,
        qformer_compressor_cfg,
        qformer_model,
        **kwargs
    ):
        super().__init__()

        print("Loading Qformer visual encoder...")
        self._vis_processor = vis_processor

        print("Loading VIT...")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_backbone:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")

        print("Loading Q-Former")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=qformer_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info("Loading Q-Former Done")

        # check if we will compress q former tokens through a perceiver
        self.qformer_compressor_cfg = qformer_compressor_cfg
        if qformer_compressor_cfg is not None:
            print("Loading Qformer Compression Perceiver")
            self.qformer_compressor = Perceiver(
                input_channels=self.Qformer.config.hidden_size,
                input_axis=1,
                num_freq_bands=64,
                max_freq=64.0,
                depth=qformer_compressor_cfg.depth,
                num_latents=qformer_compressor_cfg.num_latents,
                latent_dim=self.Qformer.config.hidden_size,
                cross_heads=qformer_compressor_cfg.cross_heads,
                latent_heads=qformer_compressor_cfg.latent_heads,
                cross_dim_head=qformer_compressor_cfg.cross_dim_head,
                latent_dim_head=qformer_compressor_cfg.latent_dim_head,
                final_classifier_head=False,
                attn_dropout=0.1,
                ff_dropout=0.1,
                weight_tie_layers=False,
                fourier_encode_data=True,
                self_per_cross_attn=qformer_compressor_cfg.self_per_cross_attn,
            )

    def embed_obs(self, image):
        """expects tensor of shape [(b t) c h w]"""
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            q_hidden_state = query_output.last_hidden_state

            # check if compressor exists
            if self.qformer_compressor_cfg is not None:
                q_hidden_state = self.qformer_compressor(q_hidden_state)

        return q_hidden_state, image_atts

    @property
    def hidden_size(self):
        return self.Qformer.config.hidden_size

    @property
    def vis_processor(self):
        return self._vis_processor

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert (
            model_name == "eva_clip_g"
        ), "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @property
    def num_tokens(self):
        if self.qformer_compressor_cfg is not None:
            return self.qformer_compressor_cfg.num_latents
        else:
            return 32


class VC1VisualProcessor:
    def __init__(self, transform):
        self._transform = transform

    def transform(self, imgs):
        imgs = einops.rearrange(imgs, "c b h w -> b c h w")
        imgs = self._transform(imgs)
        imgs = einops.rearrange(imgs, "b c h w -> c b h w")
        return imgs


class VC1ObservationEncoder(ObservationEncoder):
    def __init__(
        self, vit_precision, vit_model, freeze_backbone, max_batch_size, *args, **kwargs
    ):
        super().__init__()

        print("Loading VC1 visual encoder...")
        model, hidden_dim, transforms, _ = vc_model_utils.load_model(vit_model)
        self.backbone = model
        torch.compile(self.backbone)

        self.preprocess_transform = VC1VisualProcessor(transforms)
        self._hidden_dim = hidden_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if vit_precision == "fp16":
            convert_weights_to_fp16(self.backbone)

    def embed_obs(self, img):
        with self.maybe_autocast():
            img = img.to(self.device)
            out = self.backbone(img)
            image_embeds = einops.rearrange(out, "b h -> b 1 h")
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=self.device
            )

        return image_embeds, image_atts

    @property
    def vis_processor(self):
        return self._vis_processor

    @property
    def hidden_size(self):
        return self._hidden_dim

    @property
    def num_tokens(self):
        return 1


class CLIPObservationEncoder(ObservationEncoder):
    def __init__(
        self,
        vit_precision,
        vit_model,
        freeze_backbone,
        max_batch_size,
        fuse_rgb_goal,
        precomputed_embeddings,
        *args,
        **kwargs
    ):
        super().__init__()

        print("Loading CLIP visual encoder...")
        self.max_batch_size = max_batch_size
        self.fuse_rgb_goal = fuse_rgb_goal
        self.precomputed_embeddings = precomputed_embeddings

        self.backbone = CLIPVisionModel.from_pretrained(vit_model)
        self.preprocess_transform = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=(0, 0, 0), std=(255, 255, 255), inplace=True),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    inplace=True,
                ),
            ]
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if vit_precision == "fp16":
            convert_weights_to_fp16(self.backbone)
        # compile doesn't seem to work with bfloat16 for some reason
        # self.backbone = torch.compile(self.backbone)

        if fuse_rgb_goal:
            self.num_patches = self.backbone.vision_model.embeddings.num_patches
            num_compression_channels = int((self.hidden_size) / self.num_patches)

            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.hidden_size * 2,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
                nn.Flatten(),
            )

    def embed_obs(self, list_of_episodes):
        """
        obs is a TensorDict with keys rgb, depth, imagegoal
        """

        if not self.precomputed_embeddings:
            rgbs_l, goals_l = map(
                lambda k: [
                    einops.rearrange(episode[k], "t h w c -> t c h w")
                    for episode in list_of_episodes
                ],
                ("rgb", "imagegoal"),
            )
            rgbs_t, goals_t = map(lambda t: torch.cat(t, dim=0), (rgbs_l, goals_l))

            # compute embeddings for goals
            x = torch.cat([goals_t, rgbs_t], dim=0).float()
            x = self.preprocess_transform(x)

            with self.maybe_autocast():
                x = list(
                    map(
                        lambda i: self.backbone(
                            pixel_values=x[i : i + self.max_batch_size].to(self.device)
                        ),
                        range(0, x.shape[0], self.max_batch_size),
                    )
                )
            x = torch.cat([t.last_hidden_state for t in x])
            goals_e, rgbs_e = x[: goals_t.shape[0]], x[goals_t.shape[0] :]
            rgbs_e = torch.split(rgbs_e, [len(t) for t in rgbs_l])
        else:
            rgbs_e, goals_e = map(
                lambda k: [episode[k] for episode in list_of_episodes],
                ("rgb", "imagegoal"),
            )

        goals_out = [goals_e[i, 0:1, None] for i in range(len(list_of_episodes))]
        if self.fuse_rgb_goal:
            # fuse rgbs with goal
            rgbs_fused = [
                torch.cat(
                    [
                        rgbs_e[i],
                        einops.repeat(goals_e[i], "... -> t ...", t=rgbs_e[i].shape[0]),
                    ],
                    dim=2,
                )
                for i in range(len(rgbs_e))
            ]
            rgbs_fused = torch.cat(rgbs_fused, dim=0)
            rgbs_fused = rgbs_fused[:, 1:]
            rgbs_fused = einops.rearrange(
                rgbs_fused,
                "n (p q) c -> n c p q",
                p=int(self.num_patches**0.5),
                q=int(self.num_patches**0.5),
            )

            rgbs_fused = self.compression(rgbs_fused)[:, None, :]
            rgbs_fused = torch.split(rgbs_fused, [len(t) for t in rgbs_l])
            goals_out = [goals_e[i, 0:1, None] for i in range(len(goals_l))]
            return goals_out, rgbs_fused
        else:
            goals_e = [goals_e[i, 0:1, None] for i in range(len(goals_l))]
            return goals_e, rgbs_e

    @property
    def vis_processor(self):
        return self._vis_processor

    @property
    def hidden_size(self):
        return self.backbone.vision_model.config.hidden_size

    @property
    def num_tokens(self):
        return 1
