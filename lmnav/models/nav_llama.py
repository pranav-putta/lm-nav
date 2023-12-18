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

from lmnav.common.utils import convert_weights_to_fp16
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

    @property
    def hidden_size(self):
        return self.llama_model.config.hidden_size

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def tokenize_actions(self, actions):
        action_tkns = [
            self.llama_tokenizer(
                " ".join(act), return_tensors="pt", add_special_tokens=False
            )
            for act in actions
        ]
        action_tkns_t = torch.stack(
            [tkn.input_ids.to(self.device) for tkn in action_tkns]
        )
        return action_tkns_t

    def embed_actions(self, actions):
        action_idx2token = {0: "stop", 1: "forward", 2: "left", 3: "right"}

        # process actions into tokens
        actions = actions.tolist()
        actions = [[action_idx2token[act] for act in acts_t] for acts_t in actions]

        action_tkns_t = self.tokenize_actions(actions)

        action_embds = [
            self.llama_model.get_input_embeddings()(tkns_t) for tkns_t in action_tkns_t
        ]
        action_embds = torch.cat(action_embds, dim=0)
        return action_embds, action_tkns_t

    def embed_visual(self, rgbs, goals, precomputed_embeddings=False):
        rgbs, goals = self.vis_encoder.embed_obs(rgbs, goals, precomputed_embeddings)
        rgbs = self.llama_proj(rgbs)
        goals = self.llama_proj(goals)
        return rgbs, goals

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

        act_tgts = action_tkns_t.permute(0, 2, 1)
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

    def forward_with_embds(self, rgb_embds, goal_embds, actions_t, mask_t):
        return self.forward(rgb_embds, goal_embds, actions_t, mask_t, True)

    def forward(self, rgbs_t, goals_t, actions_t, mask_t, precomputed_embeddings=False):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """
        with self.maybe_autocast():
            rgbs_t, goals_t, mask_t, actions_t = map(
                lambda t: t.to(self.device), (rgbs_t, goals_t, mask_t, actions_t)
            )
            actions_t = actions_t.long()

            if not precomputed_embeddings:
                rgbs_embd, goals_embd = self.embed_visual(rgbs_t, goals_t, precomputed_embeddings)
            else:
                rgbs_embd, goals_embd = rgbs_t, goals_t

            mask_t = mask_t.to(torch.bool)

            embd, tgts = self.prompt1_wrap(
                self.prompt1, rgbs_embd, goals_embd, actions_t, mask_t
            )
            embd = embd.to(self.device)
            tgts = tgts.to(self.device)
            outputs = self.llama_model(
                inputs_embeds=embd, labels=tgts, return_dict=True, output_hidden_states=True
            )

            # extract the action tokens
            act_tkn_ids = self.llama_tokenizer(
                "stop forward left right", add_special_tokens=False, return_tensors="pt"
            )
            act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()
            B, T, *_ = rgbs_embd.shape

            act_positions = torch.tensor(
                [(self.tokens_per_img + 1) * (T - i - 1) + 2 for i in range(T)]
            ).to(self.device)

            act_hidden_states = outputs.hidden_states[-1][:, -act_positions]

            # construct action logits
            if self.action_head_mode == "act_linear":
                act_logits = self.action_head(act_hidden_states)
            elif self.action_head_mode == "lm" or self.action_head_mode == "lm_slice":
                act_logits = outputs.logits[:, -act_positions][:, :, act_tkn_ids]
            else:
                print(f"{self.action_head_mode} head mode is not recognized")
                exit()

            probs = F.softmax(act_logits, dim=-1)

            targets = actions_t.detach().clone().masked_fill_(~mask_t, -100)

            # compute loss
            if self.action_head_mode == "act_linear" or self.action_head_mode == "lm_slice":
                loss = F.cross_entropy(
                    einops.rearrange(probs, "b t h -> (b t) h"),
                    einops.rearrange(targets, "b t -> (b t)"),
                    ignore_index=-100,
                )
            else:
                loss = outputs.loss

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

    def fast_action_generator(self, rollout_storage, sampler, use_cache=True, max_its=None):
        """
        action generator function, takes in the next rgb, goal, and action
        """

        past_kv_cache = None
        start_token_idx = [0] * rollout_storage.num_envs
        its, seq_len = 0, 0

        while True:
            dones = yield
            
            # construct embedding for the current step
            if use_cache:
                rgb_embd = rollout_storage.rgbs[:, its]
                goal_embd = rollout_storage.goals[:, its]
                act_embd, _ = self.embed_actions(rollout_storage.actions[:, its - 1][..., None])
                embd = torch.cat([act_embd, rgb_embd], dim=1)

                # compute the attention mask and position ids based on the token start indices
                attn_mask = torch.ones(embd.shape[0], seq_len + embd.shape[1]).to(self.device)
                for i, t in enumerate(start_token_idx):
                    attn_mask[i, :t] = 0
                pos_ids = [torch.tensor(list(range(seq_len - t, seq_len + embd.shape[1] - t))) for t in start_token_idx]
                pos_ids = torch.stack(pos_ids).to(self.device)

                # forward pass
                outputs = self.llama_model(
                    inputs_embeds=embd,
                    return_dict=True,
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
                reset_rgb_embds = rgb_embd[reset_idx][:, None]
                reset_goal_embds = goal_embd[reset_idx][:, None]
                actions_t = torch.zeros(reset_rgb_embds.shape[0], 1).long().to(self.device)
                mask_t = torch.ones_like(actions_t, dtype=torch.bool).to(self.device)

                reset_embd, _ = self.prompt1_wrap(
                    self.prompt1, reset_rgb_embds, reset_goal_embds, actions_t, mask_t
                )
                reset_embd = reset_embd[:, :-1] # last token is dummy action

                # update the past key value cache
                reset_outputs = self.llama_model(
                    inputs_embeds=reset_embd,
                    return_dict=True
                )
                reset_past_kv_cache = reset_outputs.past_key_values

                new_seq_len = max(past_kv_cache[0][0].shape[2], reset_embd.shape[1])

                # left pad kv cache to match the new seq len
                pad_len = max(0, new_seq_len - past_kv_cache[0][0].shape[2])
                new_past_kv_cache = ()
                for j in range(len(past_kv_cache)):
                    k = torch.nn.functional.pad(past_kv_cache[j][0], (0, 0, pad_len, 0)) 
                    v = torch.nn.functional.pad(past_kv_cache[j][1], (0, 0, pad_len, 0))
                    new_past_kv_cache += ((k, v),)
                past_kv_cache = new_past_kv_cache

                # replace the kv cache with the reset kv cache
                pad_len = max(0, new_seq_len - reset_past_kv_cache[0][0].shape[2])
                for (i, env), j in itertools.product(enumerate(reset_idx), range(len(past_kv_cache))):
                    past_kv_cache[j][0][env] = torch.nn.functional.pad(reset_past_kv_cache[j][0][i], (0, 0, pad_len, 0)) 
                    past_kv_cache[j][1][env] = torch.nn.functional.pad(reset_past_kv_cache[j][1][i], (0, 0, pad_len, 0))

                # update start idxs
                start_token_idx = [min(0, new_seq_len - reset_embd.shape[1]) + t for t in start_token_idx]
                for i, idx in enumerate(reset_idx):
                    logits[idx] = reset_outputs.logits[i, -1]
                    start_token_idx[idx] = max(0, seq_len - reset_embd.shape[1])


            act_tkn_ids = self.llama_tokenizer(
                "stop forward left right", add_special_tokens=False, return_tensors="pt"
            )
            act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()

            # project onto the action space
            actions = [sampler(logits[i, act_tkn_ids]) for i in range(rollout_storage.num_envs)]
            seq_len = past_kv_cache[0][0].shape[2]

            its += 1

            if its == max_its:
                del past_kv_cache

            yield actions



    def action_generator(self, num_envs, maximum_likelihood=False, generator=None, max_its=0):
        """
        action generator function, takes in the next rgb, goal, and action
        """

        T = self.max_trajectory_length

        episodes = [[] for _ in range(num_envs)]

        act_tkn_ids = self.llama_tokenizer(
            "stop forward left right", add_special_tokens=False, return_tensors="pt"
        )
        act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()

        its = 0
        while True:
            (rgb_embds, goal_embds), dones = yield

            if dones is None:
                episodes.clear()

            for i, episode in enumerate(episodes):
                if dones[i]:
                    episode.clear()

                episode.append(
                    {
                        "rgb_embds": rgb_embds[i],
                        "goal_embds": goal_embds[i],
                        "action": 0,
                    }
                )

            episodes = [e[-(T - 1) :] for e in episodes]

            rgb_embds, goal_embds, actions = map(
                lambda key: [[step[key] for step in episode] for episode in episodes],
                ("rgb_embds", "goal_embds", "action"),
            )

            rgb_embds, goal_embds = map(
                lambda seq: [torch.cat(t, dim=0) for t in seq],
                (rgb_embds, goal_embds),
            )
            actions = [torch.tensor(t) for t in actions]
            rgb_embds, goal_embds, actions_t = map(
                lambda t: self.pad_sequences(t, dim=0), (rgb_embds, goal_embds, actions)
            )
            goal_embds = goal_embds[:, 0:1]
            mask_t = torch.ones_like(actions_t, dtype=torch.bool).to(self.device)

            lens = [len(e) for e in episodes]
            max_len = max(lens)

            embd, tgts = self.prompt1_wrap(
                self.prompt1, rgb_embds, goal_embds, actions_t, mask_t
            )

            embd = embd.to(self.device)
            tgts = tgts.to(self.device)

            outputs = self.llama_model(
                inputs_embeds=embd, labels=tgts, return_dict=True
            )

            act_pos_delta = [
                (self.tokens_per_img + 1) * (max_len - l) + 2 for l in lens
            ]
            logits = outputs.logits

            # project onto the action space
            actions = []
            for i in range(len(episodes)):
                act_logits = logits[i, -act_pos_delta[i], act_tkn_ids]
                act_logits = F.softmax(act_logits)

                if maximum_likelihood:
                    action = act_logits.argmax().cpu().item()
                else:
                    action = torch.multinomial(act_logits, 1, generator=generator).cpu().item()
                actions.append(action)

                episodes[i][-1]["action"] = action

            its += 1

            if its == max_its:
                embd.to("cpu")
                tgts.to("cpu")
                torch.save(outputs, "slow.pt")
                episodes.clear()
            yield actions

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optim_groups =  [{"params": [p for p in self.parameters() if p.requires_grad]}]
        optim_groups = [
            {**group, "lr": learning_rate, "weight_decay": weight_decay} for group in optim_groups
        ]
        optim = torch.optim.Adam(params=optim_groups, betas=betas)
        return optim, optim_groups

