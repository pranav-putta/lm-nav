import torch
import numpy as np
import einops
from torch import nn
from lmnav.models.modeling_clip import CLIPVisionModel
from torchvision import transforms

from lmnav.models.base_model import BaseModel
from lmnav.models.eva_vit import create_eva_vit_g
from lmnav.models.blip2 import disabled_train
from lmnav.models.Qformer import BertConfig, BertLMHeadModel
from lmnav.models.perceiver import Perceiver

from lmnav.common.dist_utils import download_cached_file
from lmnav.common.utils import  is_url, convert_weights_to_fp16
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
        raise NotImplementedError("embed_obs needs to be implemented")

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
        self,
        vit_precision,
        vit_model,
        freeze_backbone=False,
        max_batch_size=2048,
        *args,
        **kwargs
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
        freeze_backbone=False,
        max_batch_size=2048,
        fuse_rgb_goal=False,
        *args,
        **kwargs
    ):
        super().__init__()

        print("Loading CLIP visual encoder...")
        self.max_batch_size = max_batch_size
        self.fuse_rgb_goal = fuse_rgb_goal

        self.preprocess_transform = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    inplace=True,
                ),
            ]
        )

        self.backbone = CLIPVisionModel.from_pretrained(vit_model, torch_dtype=(torch.bfloat16 if vit_precision == "f16" else torch.float32))
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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

            if vit_precision == "fp16":
                convert_weights_to_fp16(self.compression)

    def embed_obs(self, rgbs, goals, precomputed_embeddings=False):
        """
        obs is a TensorDict with keys rgb, depth, imagegoal
        """
        def apply_fn_in_batches(tensor, fn, max_batch_size, rebatch_fn=torch.cat):
            out = list(
                map(
                    lambda j: fn(tensor[j : j + max_batch_size]),
                    range(0, tensor.shape[0], max_batch_size),
                )
            )
            return rebatch_fn(out)

        B, T, *_ = rgbs.shape
        if not precomputed_embeddings:
            # compute embeddings for goals
            rgbs, goals = map(
                lambda t: einops.rearrange(t, "b t c h w -> (b t) c h w").float(),
                (rgbs, goals),
            )

            x = torch.cat([goals, rgbs], dim=0).cuda().to(torch.bfloat16)
            x = self.vis_processor(x)

            with self.maybe_autocast(), torch.no_grad():
                x = list(
                    map(
                        lambda i: self.backbone(
                            pixel_values=x[i : i + self.max_batch_size]),
                        range(0, x.shape[0], self.max_batch_size),
                    )
                )
            if self.fuse_rgb_goal:
                x = torch.cat([t.last_hidden_state for t in x])
            else:
                x = einops.rearrange(
                    torch.cat([t.pooler_output for t in x]), "b h -> b 1 h"
                )
            goals_e, rgbs_e = x[: goals.shape[0]], x[goals.shape[0] :]
            goals_e, rgbs_e = map(
                lambda t: einops.rearrange(t, "(b t) ... -> b t ...", t=T),
                (goals_e, rgbs_e),
            )
        else:
            rgbs_e, goals_e = rgbs, goals
            if len(rgbs_e.shape) == 3:
                # rgb and goal tensors are of shape (b, t, h), we want (b, t, q, h)
                # where q = 1
                rgbs_e = rgbs_e[:, :, None, :]
                goals_e = goals_e[:, None, None, :]
            elif not self.fuse_rgb_goal:
                rgbs_e = rgbs_e[:, :, 0:1]
                goals_e = goals_e[:, :, 0:1]

        if self.fuse_rgb_goal:
            with self.maybe_autocast():
                # fuse rgbs with goal
                rgbs_fused = torch.cat(
                    (
                        rgbs_e,
                        einops.repeat(goals_e, "b 1 ... -> b t ...", t=rgbs_e.shape[1]),
                    ),
                    dim=-1,
                )
                # squeeze batch/time dims
                rgbs_fused = einops.rearrange(rgbs_fused, "b t ... -> (b t) ...")
                rgbs_fused = rgbs_fused[:, 1:]
                rgbs_fused = einops.rearrange(
                    rgbs_fused,
                    "b (p q) h -> b h p q",
                    p=int(self.num_patches**0.5),
                    q=int(self.num_patches**0.5),
                )

                rgbs_fused = apply_fn_in_batches(
                    rgbs_fused, self.compression, max_batch_size=3096
                )[:, None, :]
                goals_e = goals_e[:, :, 0:1]
                rgbs_fused = einops.rearrange(rgbs_fused, "(b t) ... -> b t ...", b=B)

            return rgbs_fused, goals_e
        else:
            return rgbs_e, goals_e

    @property
    def vis_processor(self):
        # return lambda x: self.preprocess_transform(images=x, return_tensors="pt", device=self.device)['pixel_values']
        return self.preprocess_transform

    @property
    def hidden_size(self):
        return self.backbone.vision_model.config.hidden_size

    @property
    def num_tokens(self):
        return 1


# for legacy configs
CLIPVisualEncoder = CLIPObservationEncoder
VC1VisualEncoder = VC1ObservationEncoder
QFormerVisualEncoder = QformerObservationEncoder
