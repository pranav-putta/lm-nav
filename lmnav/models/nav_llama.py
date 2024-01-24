from dataclasses import dataclass

from habitat_baselines.utils.common import batch_obs
from lmnav.common.episode_processor import apply_transforms_images
import logging

import contextlib
import itertools

import torch.nn.functional as F

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from lmnav.common.episode_processor import apply_transforms_actions

from lmnav.common.utils import pad_along_dim, right_pad_tensor, trim_tensor, convert_weights_to_fp16, create_mask, logprobs_from_logits
from lmnav.models.blip2 import Blip2Base, disabled_train
from lmnav.models.modeling_llama import LlamaForCausalLM

# from lmnav.models.Qformer import BertEncoder
from transformers import LlamaTokenizer, BertConfig

# from transformers.models.bert.modeling_bert import BertEncoder
import einops
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


@dataclass
class NavLLAMAOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


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
        action_head_mode="lm",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.prompt1 = "You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
                       You can choose to move { left, right, forward, stop } at every step. The goal image is {}. \
                       After every image, choose the best action. {}"

        self.vis_encoder = vis_encoder
        self.vis_processor = self.vis_encoder.vis_processor
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.max_trajectory_length = max_trajectory_length
        self.action_head_mode = action_head_mode

        logging.info("Loading LLAMA Tokenizer")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llama_model, use_fast=False
        )
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        DEFAULT_IMAGE_PATCH_TOKEN = "<ImageHere>"
        self.llama_tokenizer.add_tokens(
            [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )

        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[
            DEFAULT_IMAGE_PATCH_TOKEN
        ]

        logging.info("Loading LLAMA Model")
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={"": 0},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )
            # self.llama_model = torch.compile(self.llama_model)

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info("Loading LLAMA Done")

        logging.info("Loading LLAMA proj")
        self.llama_proj = nn.Linear(
            self.vis_encoder.hidden_size, self.llama_model.config.hidden_size
        )

        if self.action_head_mode == "act_linear":
            self.action_head = nn.Linear(self.hidden_size, 4)
            convert_weights_to_fp16(self.action_head)
        elif self.action_head_mode == "lm" and self.action_head_mode == "lm_slice":
            pass

        if freeze_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info("LLAMA proj is frozen")
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info("LLAMA proj is not frozen")

        logging.info("Loading llama_proj Done")

        if lora_config is not None:
            # wrap llama model in peft model and run fine-tuning
            print("Wrapping LLaMA in LoRA params")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)

        self.act2tkn = nn.Embedding.from_pretrained(
            self.llama_tokenizer("stop forward left right", return_tensors='pt').input_ids[:, 1:].T
        ).to(self.device)
        self.act2embd = nn.Embedding.from_pretrained(
            self.llama_model.get_input_embeddings()(self.act2tkn.weight.data).squeeze(),
            freeze=True,
        )
        self.llama_cfg = self.llama_model.config

    @property
    def hidden_size(self):
        return self.llama_model.config.hidden_size

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def embed_actions(self, actions):
        # process actions into tokens
        actions = actions.long()
        action_tkns_t = self.act2tkn(actions)
        action_embds = self.act2embd(actions)
        return action_embds, action_tkns_t

    def embed_visual(self, rgbs, goals, precomputed_embeddings=False):
        return self.vis_encoder.embed_obs(rgbs, goals, precomputed_embeddings)

    def create_prompt_embd(self, prompts, goals_embd):
        B, *_ = goals_embd.shape
        prompt_embds = []
        masks = []
        for idx, prompt in enumerate(prompts):
            prompt_segs = prompt.split("{}")
            prompt_tkns = [
                self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=(i == 0))
                for i, seg in enumerate(prompt_segs)
                if len(seg)
            ]
            prompt_embd = [
                self.llama_model.get_input_embeddings()(tkns.input_ids.to(self.device))
                for tkns in prompt_tkns
            ]
            goals_embd = goals_embd[idx, 0] # goal embd token is the same, just take the first one
            embd = [prompt_embd[0][0], goals_embd, prompt_embd[1][0]]
            embd = torch.cat(embd, dim=0)
            mask = torch.ones(embd.shape[0], dtype=torch.bool).to(self.device)

            prompt_embds.append(embd)
            masks.append(mask)
        max_len = max([embd.shape[0] for embd in prompt_embds])
        prompt_embds = pad_along_dim(prompt_embds, max_len, padding_side='left').to(self.device)
        masks = pad_along_dim(masks, max_len, padding_side='left').to(self.device)
        return prompt_embds, masks

    def prompt1_wrap(self, prompt, rgbs_embd, goals_embd, actions, masks):
        actions_embd, action_tkns_t = self.embed_actions(actions)

        B, T, Q, H = rgbs_embd.shape

        prompt_segs = prompt.split("{}")
        prompt_tkns = [
            self.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=(i == 0))
            for i, seg in enumerate(prompt_segs)
            if len(seg)
        ]
        prompt_embd = [
            self.llama_model.get_input_embeddings()(tkns.input_ids.to(self.device))
            for tkns in prompt_tkns
        ]
        prompt_embd = [embd.repeat(B, 1, 1) for embd in prompt_embd]

        actions_embd = einops.rearrange(actions_embd, "b t h -> b t 1 h")

        s_a_embds = torch.cat([rgbs_embd, actions_embd], dim=2)
        s_a_embds = s_a_embds.view(B, T * (Q + 1), H)

        goals_embd = goals_embd[:, 0] # goal embd token is the same, just take the first one

        embds = [prompt_embd[0], goals_embd, prompt_embd[1], s_a_embds]
        embds = torch.cat(embds, dim=1)

        # construct targets
        prompt_tgts = (
            torch.ones(
                B,
                prompt_embd[0].shape[1] + goals_embd.shape[1] + prompt_embd[1].shape[1],
            )
            .fill_(-100)
            .to(self.device)
        )
        rgb_tgts = torch.ones(B, T, Q).fill_(-100).to(self.device)

        # act_tgts = action_tkns_t.permute(0, 2, 1)
        act_tgts = action_tkns_t
        act_tgts = act_tgts.masked_fill_(~masks[..., None], -100)

        s_a_tgts = (
            torch.cat([rgb_tgts, act_tgts], dim=2).view(B, T * (Q + 1)).to(self.device)
        )
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

    def prepare_state_action_tensor(self, obs_embds, action_embds, mask, input_format='as'):
        # interleave the rgb and action embeddings together
        b, t, q, h = obs_embds.shape
        action_embds = einops.rearrange(action_embds, 'b t h -> b t 1 h')
        if input_format == 'as':
            obs_embds = F.pad(obs_embds, (0, 0, 0, 0, 0, 1))
            embds = torch.cat([action_embds, obs_embds], dim=2)
            embds = einops.rearrange(embds, 'b t q h -> b (t q) h')[:, :-q]
            seq_lens = mask.sum(dim=1) * (q + 1) + 1 # (q + 1) for the action token, +1 for the prev action token
        elif input_format == 'sa':
            embds = torch.cat([obs_embds, action_embds], dim=2)
            embds = einops.rearrange(embds, 'b t q h -> b (t q) h')
            seq_lens = mask.sum(dim=1) * (q + 1) # (q + 1) for the action token
        else:
            raise ValueError(f"input_format={input_format} not supported")

        new_mask = create_mask(seq_lens, embds.shape[1]).to(self.device)
        return embds, new_mask

    def forward_with_embds(self, prompt, rgb_embds, goal_embds, actions_t, mask_t, pvk_t, attnm_t):
        return self.forward(prompt, rgb_embds, goal_embds, actions_t, mask_t, pvk_t, attnm_t, True)

    def forward(self, prompts, rgbs_t, goals_t, actions_t, mask_t,
                pvk_t=None, past_attnm_t=None, precomputed_embeddings=False,
                input_format='as'):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """
        with self.maybe_autocast(dtype=torch.bfloat16):
            rgbs_embd, goals_embd = self.embed_visual(rgbs_t, goals_t, precomputed_embeddings)
            rgbs_embd, goals_embd = map(lambda x: self.llama_proj(x), [rgbs_embd, goals_embd])
            action_embds = self.act2embd(actions_t)

            if past_attnm_t is not None:
                past_lengths = past_attnm_t.sum(dim=1)
            else:
                past_lengths = torch.zeros(rgbs_embd.shape[0], device=self.device)

            B, T, *_ = rgbs_embd.shape
            embds, mask = self.prepare_state_action_tensor(rgbs_embd, action_embds, mask_t, input_format=input_format)
            seq_lens = mask.sum(dim=1)

            # construct prompt embeddings for the reset episodes
            reset_episode_mask = past_lengths == 0
            prompt_embds, prompt_mask = self.create_prompt_embd(prompts, goals_embd[reset_episode_mask])
            prompt_lens = prompt_mask.sum(dim=1)
            max_seq_len = max(embds.shape[1], seq_lens.max() + prompt_embds.shape[1]) # -1 bc prev action not needed for reset episodes
            embds = F.pad(embds, (0, 0, 0, max_seq_len - embds.shape[1]))
            mask = F.pad(mask, (0, max_seq_len - mask.shape[1]))
            for i, idx in enumerate(torch.where(reset_episode_mask)[0].tolist()):
                # if input_format is 'as', then we need to offset by 1 to account for the prev action token which is not needed for reset episodes
                offset = 1 if input_format == 'as' else 0 
                pad_len = embds.shape[1] - prompt_embds[i].shape[0]
                embds[idx, :] = torch.cat([prompt_embds[i], embds[idx, offset:pad_len + offset]], dim=0)
                mask[idx, :] = torch.cat([prompt_mask[i], mask[idx, offset:pad_len + offset]], dim=0)

            embds, mask = trim_tensor(embds, mask)
            seq_lens = mask.sum(dim=1)

            embds = right_pad_tensor(embds, seq_lens)
            mask = right_pad_tensor(mask, seq_lens)

            pos_ids = None
            if pvk_t is not None and past_attnm_t is not None:
                pvk_t = pvk_t.to(self.device)
                pvk_t = einops.rearrange(pvk_t, "b l k ... -> l k b ...")
                past_attnm_t = past_attnm_t.to(self.device).long()
                pos_ids = torch.arange(embds.shape[1]).repeat((embds.shape[0], 1)).to(self.device) + past_attnm_t.sum(dim=1)[:, None]
                past_attnm_t = torch.cat([past_attnm_t, torch.ones(past_attnm_t.shape[0], embds.shape[1]).to(self.device)], dim=1)

            max_episode_len = mask_t.sum(dim=1).max() 
            # input embeddings are A S A S A ..., we want hidden states after S 
            output_idx = torch.arange(0, max_episode_len, device=self.device).repeat(B, 1) * (self.tokens_per_img + 1) + 1
            # for reset episodes, we want the hidden state after the prompt: prompt + *S* A S A ... these also don't have the prev action
            output_idx[reset_episode_mask] += prompt_lens.unsqueeze(-1) - 1
            output_idx.masked_fill_(~mask_t[:, :max_episode_len], 0)

            # if input_format is 'as', then we need to offset by 1 to account for the prev action token which is not needed for reset episodes
            offset = 1 if input_format == 'as' else 0 
            action_tkns = self.act2tkn(actions_t).squeeze(-1)
            masked_targets = action_tkns[:, offset:max_episode_len + offset].clone().masked_fill_(~mask_t[:, :max_episode_len], -100)

            targets = torch.full((embds.shape[0], embds.shape[1]), -100).to(self.device)
            targets.scatter_(1, output_idx, masked_targets)

            outputs = self.llama_model(
                inputs_embeds=embds, 
                labels=targets,
                return_dict=True,
                output_hidden_states=True,
                past_key_values=pvk_t,
                attention_mask=past_attnm_t,
                position_ids=pos_ids,
            )


            # extract the action tokens
            act_tkn_ids = self.act2tkn.weight.data.squeeze()

            output_idx_hx = output_idx[..., None].repeat(1, 1, outputs.hidden_states[-1].shape[-1])
            act_hidden_states = outputs.hidden_states[-1].gather(1, output_idx_hx)
            act_hidden_states.masked_fill_(~mask_t[:, :max_episode_len, None], 0)
            output_idx_logits = output_idx[..., None].repeat(1, 1, outputs.logits.shape[-1])
            act_logits = outputs.logits.gather(1, output_idx_logits)
            act_logits = act_logits[:, :, act_tkn_ids]
            act_logits.masked_fill_(~mask_t[:, :max_episode_len, None], 0)

            # compute loss
            loss = outputs.loss
            # pad logits and hidden states to max episode length
            original_len = mask_t.shape[1]
            act_logits = F.pad(act_logits, (0,0,0, original_len - act_logits.shape[1]))
            act_hidden_states = F.pad(act_hidden_states, (0, 0, 0, original_len - act_hidden_states.shape[1]))
            return NavLLAMAOutput(
                logits=act_logits, loss=loss, last_hidden_state=act_hidden_states
            )

    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])

        padded_seqs = [
            F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs
        ]
        return torch.stack(padded_seqs)

    @property
    def tokens_per_img(self):
        return self.vis_encoder.num_tokens

    def action_generator(self, rollouts, sampler):
        """ action generator function, takes in the next rgb, goal, and action """
        num_envs = rollouts.num_envs
        its = 0

        rollouts.embeddings = []

        # set up initial hidden states and variables
        if rollouts.last_cache is None:
            past_lens = torch.tensor([0] * rollouts.num_envs, device=self.device)
            past_kv_cache = ()
            h = self.llama_cfg.num_attention_heads
            d = self.llama_cfg.hidden_size // self.llama_cfg.num_attention_heads
            for j in range(self.llama_cfg.num_hidden_layers):
                past_kv_cache += ((torch.zeros(num_envs, h, 462, d, device=self.device, dtype=torch.bfloat16),
                                   torch.zeros(num_envs, h, 462, d, device=self.device, dtype=torch.bfloat16)),)
        else:
            past_lens = rollouts.last_cache.past_lengths.clone()
            past_kv_cache = rollouts.last_cache.past_kv_cache.clone().to(self.device)

        # loop over the rollouts
        while True:
            dones = yield
            it = rollouts.current_step_idx
            dones = torch.tensor(dones, device=self.device, dtype=torch.bool)

            # construct embedding for the current step
            rgb_embd, goal_embd = map(lambda x: self.llama_proj(x.float()), (rollouts.rgbs[:, it], rollouts.goals[:, it]))
            act_embd, _ = self.embed_actions(rollouts.prev_actions[:, it][..., None])
            embd = torch.cat([act_embd, rgb_embd], dim=1)
            next_n_tkns = embd.shape[1]
            next_lens = past_lens + next_n_tkns

            # check if we need to reset an episode and update embeddings
            if dones.any():
                # construct prompt embeddings for the reset episodes
                reset_rgb_embd = rgb_embd[dones][:, None]
                reset_goal_embd = goal_embd[dones][:, None]
                reset_act_embd = torch.zeros(reset_rgb_embd.shape[0], 1).long().to(self.device)
                reset_mask_t = torch.ones_like(reset_act_embd, dtype=torch.bool).to(self.device)
                reset_embd, _ = self.prompt1_wrap(
                    self.prompt1, reset_rgb_embd, reset_goal_embd, reset_act_embd, reset_mask_t
                )
                reset_embd = reset_embd[:, :-1] # last token is dummy action

                # update the input embeddings 
                # 1. expand the embedding to match the reset embedding length
                embd = F.pad(embd, (0, 0, 0, reset_embd.shape[1] - embd.shape[1]))
                # 2. replace the reset embeddings
                embd[dones] = reset_embd
                # 3. update episode lengths
                past_lens[dones] = 0
                next_lens[dones] = reset_embd.shape[1]

            # compute the attention mask and position ids based on the token start indices
            cache_len = past_kv_cache[0][0].shape[2]

            attn_mask = F.pad(create_mask(past_lens, cache_len), (0, embd.shape[1]), mode='constant', value=1).to(self.device)
            pos_ids = torch.arange(0, embd.shape[1], device=self.device).repeat(num_envs, 1) + past_lens.unsqueeze(-1)
            output_idx = next_lens - past_lens - 1
            
            # run forward pass
            outputs = self.llama_model(
                inputs_embeds=embd,
                past_key_values=past_kv_cache,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                return_dict=True,
                output_hidden_states=True,
            )


            rollouts.embeddings.append(embd[:, :2])
            assert output_idx.max() < outputs.logits.shape[1], f"output_idx.max()={output_idx.max()}, logits.shape[1]={outputs.logits.shape[1]}"
            logits = outputs.logits[torch.arange(num_envs), output_idx]
            past_kv_cache = outputs.past_key_values
            hidden_state = outputs.hidden_states[-1][torch.arange(num_envs), output_idx]

            # sample actions
            logits = logits[:, self.act2tkn.weight.data.squeeze()]
            actions = list(map(lambda i: sampler(logits[i]), range(num_envs)))
            logprobs = logprobs_from_logits(logits[:, None, :], torch.tensor(actions, device=self.device, dtype=torch.long)[:, None]).squeeze(1)
            past_lens = next_lens

            if its == rollouts.max_steps - 1:
                rollouts.set_current_cache(past_lens, past_kv_cache)

            its += 1
            yield actions, logprobs, hidden_state



    def fast_action_generator(self, rollout_storage, sampler, use_cache=True, max_its=100):
        """
        action generator function, takes in the next rgb, goal, and action
        """

        start_token_idx, past_kv_cache = rollout_storage.last_cache
        num_envs = rollout_storage.num_envs
        if start_token_idx is None:
            start_token_idx = torch.tensor([0] * num_envs)
        if past_kv_cache is not None:
            new_past_kv_cache = () 
            for j in range(len(past_kv_cache)):
                new_past_kv_cache += ((past_kv_cache[j][0].to(self.device), past_kv_cache[j][1].to(self.device)),)
            past_kv_cache = new_past_kv_cache
            seq_len = past_kv_cache[0][0].shape[2]
        else:
            seq_len = 0

        while True:
            dones = yield
            its = rollout_storage.current_step_idx

            # construct embedding for the current step
            if use_cache:
                # construct the embeddings for the next rgb and goal step
                rgb_embd = rollout_storage.rgbs[:, its]
                goal_embd = rollout_storage.goals[:, its]
                act_embd, _ = self.embed_actions(rollout_storage.actions[:, its - 1][..., None])
                embd = torch.cat([act_embd, rgb_embd], dim=1)

                # compute the attention mask and position ids based on the token start indices
                attn_mask = ~create_mask(start_token_idx, seq_len + embd.shape[1]).to(self.device)
                pos_ids = torch.arange(seq_len, seq_len + embd.shape[1]).repeat(num_envs, 1) - start_token_idx.unsqueeze(-1)
                pos_ids = pos_ids.to(self.device)

                # forward pass
                outputs = self.llama_model(
                    inputs_embeds=embd,
                    past_key_values=past_kv_cache,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                )
                logits = outputs.logits[:, -1]
                past_kv_cache = outputs.past_key_values
            else:
                start_indices = torch.argmax(torch.cumsum(rollout_storage.dones[:, :its+1], dim=1), dim=1)
                fn_pad_embd = lambda embd, start, dim: torch.nn.functional.pad(embd, (0,) * ((len(embd.shape) - dim - 1) * 2 + 1) + (start,), value=0)
                rgb_embd = torch.stack([fn_pad_embd(rollout_storage.rgbs[i, start:its+1], start, 0) for i, start in enumerate(start_indices)])
                goal_embd = torch.stack([fn_pad_embd(rollout_storage.goals[i, start:its+1], start, 0) for i, start in enumerate(start_indices)])
                act_embd = torch.stack([fn_pad_embd(rollout_storage.actions[i, start:its+1], start, 0) for i, start in enumerate(start_indices)])
                mask_t = torch.ones_like(act_embd, dtype=torch.bool).to(self.device)

                embd, _ = self.prompt1_wrap(
                    self.prompt1, rgb_embd, goal_embd, act_embd, mask_t
                )
                embd = embd[:, :-1] # last token is dummy action
                outputs = self.llama_model(
                    inputs_embeds=embd,
                    return_dict=True,
                )

                logits = outputs.logits[range(outputs.logits.shape[0]), start_indices - 1]
                past_kv_cache = outputs.past_key_values

            # recompute indices that are done and re-populate the cache
            reset_idx = torch.tensor([i for i, done in enumerate(dones) if done]).long()
            if len(reset_idx) > 0 and use_cache:
                # first compute the reset embeddings
                rgb_embd = rgb_embd[reset_idx][:, None]
                goal_embd = goal_embd[reset_idx][:, None]
                act_embd = torch.zeros(rgb_embd.shape[0], 1).long().to(self.device)
                mask_t = torch.ones_like(act_embd, dtype=torch.bool).to(self.device)

                embd, _ = self.prompt1_wrap(
                    self.prompt1, rgb_embd, goal_embd, act_embd, mask_t
                )
                embd = embd[:, :-1] # last token is dummy action

                # update the past key value cache
                outputs = self.llama_model(inputs_embeds=embd)
                reset_past_kv_cache = outputs.past_key_values
                new_seq_len = max(past_kv_cache[0][0].shape[2], embd.shape[1])

                # left pad kv cache to match the new seq len
                pad_len = max(0, new_seq_len - past_kv_cache[0][0].shape[2])
                new_past_kv_cache = ()
                for j in range(len(past_kv_cache)):
                    k = torch.nn.functional.pad(past_kv_cache[j][0], (0, 0, pad_len, 0)) 
                    v = torch.nn.functional.pad(past_kv_cache[j][1], (0, 0, pad_len, 0))
                    new_past_kv_cache += ((k, v),)
                del past_kv_cache
                past_kv_cache = new_past_kv_cache

                # replace the kv cache with the reset kv cache
                pad_len = max(0, new_seq_len - reset_past_kv_cache[0][0].shape[2])
                for (i, env), j in itertools.product(enumerate(reset_idx), range(len(past_kv_cache))):
                    past_kv_cache[j][0][env] = torch.nn.functional.pad(reset_past_kv_cache[j][0][i], (0, 0, pad_len, 0)) 
                    past_kv_cache[j][1][env] = torch.nn.functional.pad(reset_past_kv_cache[j][1][i], (0, 0, pad_len, 0))
                del reset_past_kv_cache

                # update start idxs
                start_token_idx = torch.tensor([min(0, new_seq_len - embd.shape[1]) + t for t in start_token_idx])
                for i, idx in enumerate(reset_idx):
                    logits[idx] = outputs.logits[i, -1]
                    start_token_idx[idx] = max(0, new_seq_len - embd.shape[1])

                # if there's empty history trim the past key value cache
                earliest_start_idx = min(start_token_idx)
                if earliest_start_idx > 0:
                    new_past_kv_cache = ()
                    for j in range(len(past_kv_cache)):
                        k = past_kv_cache[j][0].narrow(2, earliest_start_idx, past_kv_cache[j][0].shape[2] - earliest_start_idx)
                        v = past_kv_cache[j][1].narrow(2, earliest_start_idx, past_kv_cache[j][1].shape[2] - earliest_start_idx)
                        new_past_kv_cache += ((k, v),)
                    del past_kv_cache
                    past_kv_cache = new_past_kv_cache
                    for env in range(len(start_token_idx)):
                        start_token_idx[env] -= earliest_start_idx

                del outputs
                del goal_embd

            act_tkn_ids = self.llama_tokenizer(
                "stop forward left right", add_special_tokens=False, return_tensors="pt"
            )
            act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()

            # project onto the action space
            actions = [sampler(logits[i, act_tkn_ids]) for i in range(rollout_storage.num_envs)]
            seq_len = past_kv_cache[0][0].shape[2] 

            if its == max_its - 1:
                # save cache into rollout storage
                new_past_kv_cache = ()
                for j in range(len(past_kv_cache)):
                    new_past_kv_cache += ((past_kv_cache[j][0].cpu(), past_kv_cache[j][1].cpu()),)
                del past_kv_cache
                past_kv_cache = new_past_kv_cache
                rollout_storage.current_hidden_state = (start_token_idx, past_kv_cache)

            with open("acts2.txt", "a+") as f:
                f.write(str(logits[0, act_tkn_ids]) + "\n")
            yield actions

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optim_groups =  [{"params": [p for p in self.parameters() if p.requires_grad]}]
        optim_groups = [
            {**group, "lr": learning_rate, "weight_decay": weight_decay} for group in optim_groups
        ]
        optim = torch.optim.Adam(params=optim_groups, betas=betas)
        return optim, optim_groups

