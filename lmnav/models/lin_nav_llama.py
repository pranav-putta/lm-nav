import logging
import random

import torch.nn.functional as F

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from lmnav.common.episode_processor import apply_transforms_inputs, extract_inputs_from_dataset

from lmnav.common.registry import registry
from lmnav.models.blip2 import Blip2Base, disabled_train
from lmnav.models.modeling_llama import LlamaForCausalLM
# from lmnav.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from lmnav.models.Qformer import BertConfig, BertLMHeadModel
from lmnav.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from lmnav.models.ImageBind.models import imagebind_model

# from flamingo_pytorch import PerceiverResampler
@registry.register_model("lin_nav_llama")
class LinNavLLAMA(Blip2Base):
    """
    Extension from BLIP2 GPT-LLAMA model to operate over navigation space.
    Adds a linear transformation to the BLIP projections.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        frozen_llama_proj=True,
        llama_proj_model='',
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = model.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_Qformer_visual(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            q_hidden_state = query_output.last_hidden_state
            
            inputs_llama = self.llama_proj(q_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
            
        inputs_llama = einops.rearrange(inputs_llama, '(b t) q h -> b t q h', b=batch_size)
        atts_llama = einops.rearrange(atts_llama, '(b t) h -> b t h', b=batch_size)
        return inputs_llama, atts_llama
    
   
    def embed_visual(self, imgs):
        img_embds, img_attns = self.encode_Qformer_visual(imgs.half().to(self.device))
        return img_embds, img_attns

    def tokenize_actions(self, actions):
        action_tkns = [self.llama_tokenizer(' '.join(act), return_tensors='pt', add_special_tokens=False) for act in actions]
        action_tkns_t = torch.stack([tkn.input_ids.to(self.device) for tkn in action_tkns])
        return action_tkns_t


    def embed_actions(self, action_tkns_t):
        action_embds = [self.llama_model.model.embed_tokens(tkns_t) for tkns_t in action_tkns_t]
        action_embds = torch.cat(action_embds, dim=0)
        return action_embds


    def prompt1_wrap(self, prompt, rgbs, goals, actions):
        rgbs_embd, rgbs_attn = self.embed_visual(rgbs)
        goals_embd, goals_attn = self.embed_visual(goals)
        action_tkns_t = self.tokenize_actions(actions)
        actions_embd = self.embed_actions(action_tkns_t)

        
        B, T, Q, H = rgbs_embd.shape
     
        prompt_segs = prompt.split("{}")
        prompt_tkns = [self.llama_tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)) for i, seg in enumerate(prompt_segs) if len(seg)]
        prompt_embd = [self.llama_model.model.embed_tokens(tkns.input_ids.to(self.device)) for tkns in prompt_tkns]
        prompt_embd = [embd.repeat(B, 1, 1) for embd in prompt_embd]
        
        actions_embd = einops.rearrange(actions_embd, 'b t h -> b t 1 h')
        s_a_embds = torch.cat([rgbs_embd, actions_embd], dim=2)
        s_a_embds = s_a_embds.view(B, T * (Q + 1), H)

        goals_embd = goals_embd.squeeze(1)

        embds = [prompt_embd[0], goals_embd, prompt_embd[1], s_a_embds]
        embds = torch.cat(embds, dim=1)

        # construct targets
        prompt_tgts = torch.ones(B, prompt_embd[0].shape[1] + goals_embd.shape[1] + prompt_embd[1].shape[1]).fill_(-100).to(self.device)
        rgb_tgts = torch.ones(B, T, Q).fill_(-100).to(self.device)
        act_tgts = action_tkns_t.permute(0, 2, 1)
        s_a_tgts = torch.cat([rgb_tgts, act_tgts], dim=2).view(B, T * (Q + 1)).to(self.device)

        tgts = torch.cat([prompt_tgts, s_a_tgts], dim=1).long().to(self.device) 
        
        return embds, tgts


    def forward(self, rgbs_t, goals_t, actions_t):
        """
        batch = {
            'rgbs': torch.Tensor[B, T, C, H, W]
            'goals': torch.Tensor[B, 1, C, H, W]
            'actions': torch.Tensor[B, T]
            }
        """
        
        im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
        prompt1 = "You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
           You can choose to move { left, right, forward, stop } at every step. The goal image is {}. \
           After every image, choose the best action. {}"
        
        embd, tgts = self.prompt1_wrap(prompt1, rgbs_t, goals_t, actions_t)
        embd = embd.to(self.device)
        tgts = tgts.to(self.device)

        outputs = self.llama_model(inputs_embeds=embd,
                                    labels=tgts,
                                    return_dict=True)
        
        return outputs

    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])
        
        padded_seqs = [F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs]
        return torch.stack(padded_seqs)


    def action_generator(self, num_envs, T, vis_processor, deterministic=False):
        """
            action generator function, takes in the next rgb, goal, and action 
        """
        
        episodes = [ [] for _ in range(num_envs) ] 
        
        act_tkn_ids = self.llama_tokenizer('stop forward left right', add_special_tokens=False, return_tensors='pt') 
        act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()
        
        while True:
            observations, episode_idxs_to_reset = yield
            
            for i, episode in enumerate(episodes):
                if i in episode_idxs_to_reset:
                    episode.clear()

                episode.append({
                                    'observation': observations[i],
                                    'action': 0
                                })

            
            partial_episodes = [e[-(T - 1):] for e in episodes]
            rgbs, goals, actions = extract_inputs_from_dataset(partial_episodes)

            rgbs_t = self.pad_sequences(rgbs, dim=0)
            actions_t = self.pad_sequences(actions, dim=0)
            goals_t = self.pad_sequences(goals, dim=0)
            lens = [len(e) for e in partial_episodes]
            max_len = max(lens)

            rgbs_t, goals_t, actions_t = apply_transforms_inputs(vis_processor, rgbs_t, goals_t, actions_t)
            outputs = self(rgbs_t, goals_t, actions_t) 

            act_pos_delta = [33 * (max_len - l) + 2 for l in lens]
            logits = outputs.logits

            # project onto the action space
            actions = []
            for i in range(len(partial_episodes)):
                act_logits = logits[i, -act_pos_delta[i], act_tkn_ids]
                act_logits = F.softmax(act_logits)

                action = act_logits.argmax().cpu().item()
                actions.append(action)

                episodes[i][-1]['action'] = action

            yield actions
            
       

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            frozen_llama_proj=frozen_llama_proj,
            llama_proj_model=llama_proj_model,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
