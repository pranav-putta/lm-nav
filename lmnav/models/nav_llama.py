import logging

import pdb
import contextlib
import random
import time

import torch.nn.functional as F

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from lmnav.common.episode_processor import apply_transforms_actions, apply_transforms_images, apply_transforms_inputs, extract_inputs_from_dataset

from lmnav.common.registry import registry
from lmnav.models.blip2 import Blip2Base, disabled_train
from lmnav.models.modeling_llama import LlamaForCausalLM
# from lmnav.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class NavLLAMA(Blip2Base):
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
        vis_encoder,
        freeze_llama_proj,
        low_resource,
        lora_config,
        llama_model,
        max_trajectory_length,
        **kwargs
    ):
        super().__init__()
        
        self.prompt1 = "You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
                       You can choose to move { left, right, forward, stop } at every step. The goal image is {}. \
                       After every image, choose the best action. {}"

        self.vis_encoder = vis_encoder
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.max_trajectory_length = max_trajectory_length
           
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
                device_map={'': 0}
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
            self.vis_encoder.hidden_size, self.llama_model.config.hidden_size
        )

        if freeze_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        if lora_config is not None:
            # wrap llama model in peft model and run fine-tuning
            print("Wrapping LLaMA in LoRA params")
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                     inference_mode=False,
                                     r=lora_config.rank,
                                     lora_alpha=lora_config.alpha,
                                     lora_dropout=lora_config.dropout)
            self.llama_model = get_peft_model(self.llama_model, peft_config)
                   

    @property
    def hidden_size(self):
        return self.llama_model.config.hidden_size
    
    
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
    

    def tokenize_actions(self, actions):
        action_tkns = [self.llama_tokenizer(' '.join(act), return_tensors='pt', add_special_tokens=False) for act in actions]
        action_tkns_t = torch.stack([tkn.input_ids.to(self.device) for tkn in action_tkns])
        return action_tkns_t


    def embed_actions(self, action_tkns_t):
        action_embds = [self.llama_model.get_input_embeddings()(tkns_t) for tkns_t in action_tkns_t]
        action_embds = torch.cat(action_embds, dim=0)
        return action_embds

    def embed_visual(self, imgs):
        B, _, T, _, _ = imgs.size()
        image = einops.rearrange(imgs, 'b c t h w -> (b t) c h w')
        image_embeds, image_atts = self.vis_encoder.embed_visual(image)
        
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)

        inputs_llama = einops.rearrange(inputs_llama, '(b t) q h -> b t q h', b=B)
        atts_llama = einops.rearrange(atts_llama, '(b t) h -> b t h', b=B)

        return inputs_llama, atts_llama

    
    def prompt1_wrap(self, prompt, rgbs_embd, goals_embd, actions, masks):
        action_tkns_t = self.tokenize_actions(actions)
        actions_embd = self.embed_actions(action_tkns_t)
        
        B, T, Q, H = rgbs_embd.shape
     
        prompt_segs = prompt.split("{}")
        prompt_tkns = [self.llama_tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)) for i, seg in enumerate(prompt_segs) if len(seg)]
        prompt_embd = [self.llama_model.get_input_embeddings()(tkns.input_ids.to(self.device)) for tkns in prompt_tkns]
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
        act_tgts = act_tgts.masked_fill_(~masks[..., None], -100)

        s_a_tgts = torch.cat([rgb_tgts, act_tgts], dim=2).view(B, T * (Q + 1)).to(self.device)
        tgts = torch.cat([prompt_tgts, s_a_tgts], dim=1).long().to(self.device) 
        
        return embds, tgts


    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    

    def forward_with_embds(self, rgb_embds, goal_embds, actions_t, mask_t):
        rgb_t, goal_t, mask_t = map(lambda t: t.to(self.device), (rgb_embds, goal_embds, mask_t))
        
        mask_t = mask_t.to(torch.bool)
        
        actions_t = apply_transforms_actions(actions_t)
        embd, _ = self.prompt1_wrap(self.prompt1, rgb_t, goal_t, actions_t, mask_t)
        outputs = self.llama_model(inputs_embeds=embd,
                                    return_dict=True,
                                    output_hidden_states=True)

        return outputs

            
    def forward(self, rgbs_t, goals_t, actions_t, mask_t):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """
        # with self.maybe_autocast():
        rgbs_t, goals_t, mask_t = map(lambda t: t.to(self.device), (rgbs_t, goals_t, mask_t))
        rgbs_embd, rgbs_attn = self.embed_visual(rgbs_t)
        goals_embd, goals_attn = self.embed_visual(goals_t)
        mask_t = mask_t.to(torch.bool)
        
        embd, tgts = self.prompt1_wrap(self.prompt1, rgbs_embd, goals_embd, actions_t, mask_t)
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


    @property
    def tokens_per_img(self):
        return self.vis_encoder.num_tokens

    def action_generator(self, num_envs, deterministic=False, max_its=0):
        """
            action generator function, takes in the next rgb, goal, and action 
        """
        T = self.max_trajectory_length
        vis_processor = self.vis_encoder.vis_processor
        
        episodes = [ [] for _ in range(num_envs) ] 
        
        act_tkn_ids = self.llama_tokenizer('stop forward left right', add_special_tokens=False, return_tensors='pt') 
        act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()
        
        its = 0
        while True:
            (rgb_embds, goal_embds), dones = yield
            if dones is None:
                episodes.clear()
            
            for i, episode in enumerate(episodes):
                if dones[i]:
                    episode.clear()

                episode.append({
                                'rgb_embds': rgb_embds[i],
                                'goal_embds': goal_embds[i],
                                'action': 0
                            })

            episodes = [e[-(T - 1):] for e in episodes]

            rgb_embds, goal_embds, actions = map(lambda key: [[step[key] for step in episode] for episode in episodes], ('rgb_embds', 'goal_embds', 'action'))
            rgb_embds, goal_embds = map(lambda seq: [torch.stack(t, dim=0) for t in seq], (rgb_embds, goal_embds))
            actions = [torch.tensor(t) for t in actions]
            rgb_embds, goal_embds, actions_t = map(lambda t: self.pad_sequences(t, dim=0), (rgb_embds, goal_embds, actions))
            goal_embds = goal_embds[:, 0:1] 
            mask_t = torch.ones_like(actions_t, dtype=torch.bool).to(self.device)

            actions_t = apply_transforms_actions(actions_t)
            
            lens = [len(e) for e in episodes]
            max_len = max(lens)
            
            embd, tgts = self.prompt1_wrap(self.prompt1, rgb_embds, goal_embds, actions_t, mask_t)
            
            embd = embd.to(self.device)
            tgts = tgts.to(self.device)
            
            outputs = self.llama_model(inputs_embeds=embd,
                                        labels=tgts,
                                        return_dict=True)

            act_pos_delta = [(self.tokens_per_img + 1) * (max_len - l) + 2 for l in lens]
            logits = outputs.logits

            # project onto the action space
            actions = []
            for i in range(len(episodes)):
                act_logits = logits[i, -act_pos_delta[i], act_tkn_ids]
                act_logits = F.softmax(act_logits)

                if deterministic:
                    action = act_logits.argmax().cpu().item()
                else:
                    action = torch.multinomial(act_logits, 1).cpu().item()
                actions.append(action)

                episodes[i][-1]['action'] = action

            its += 1
            
            
            if its == max_its:
                embd.to('cpu')
                tgts.to('cpu')
                episodes.clear() 
                torch.cuda.empty_cache()
            yield actions
            
