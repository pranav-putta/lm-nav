from dataclasses import dataclass
import pdb
from torch import nn

import math
import torch
import contextlib
import torch.nn as nn
import torch.nn.functional as F
import einops

from lmnav.models.base_model import BaseModel

from torch.cuda.amp import autocast as autocast


@dataclass
class NavGRUOutput:
    loss: torch.Tensor
    probs: torch.Tensor


class NavGRU(BaseModel):
    def __init__(
        self,
        vis_encoder,
        d_hidden,
        n_layer,
        drop_p,
        weight_decay,
        max_trajectory_length,
        *args,
        **kwargs
    ):
        super().__init__()

        self.vis_encoder = vis_encoder
        self.vis_processor = self.vis_encoder.vis_processor

        self.d_hidden = d_hidden
        self.n_layer = n_layer
        self.drop_p = drop_p
        self.weight_decay = weight_decay

        self.max_trajectory_length = max_trajectory_length
        self.max_tokens = (
            max_trajectory_length + (max_trajectory_length + 1) * self.tokens_per_img
        )  # action tokens + rgb tokens + goal token

        self.gru = nn.GRU(d_hidden, d_hidden, n_layer, batch_first=True)
        self.vis_proj = nn.Linear(self.vis_encoder.hidden_size, self.d_hidden)
        self.action_embedding = nn.Embedding(4, self.d_hidden)
        self.action_head = nn.Linear(self.d_hidden, 4)

    @property
    def hidden_size(self):
        return self.d_hidden

    @property
    def tokens_per_img(self):
        return self.vis_encoder.num_tokens

    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])

        padded_seqs = [
            F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs
        ]
        return torch.stack(padded_seqs)

    def embed_visual(self, imgs):
        with self.maybe_autocast():
            B, _, T, _, _ = imgs.size()
            imgs = imgs.to(self.device)
            image = einops.rearrange(imgs, "b c t h w -> (b t) c h w")

            image_embeds, image_atts = self.vis_encoder.embed_visual(image)
            inputs = self.vis_proj(image_embeds)

            atts = torch.ones(inputs.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            inputs = einops.rearrange(inputs, "(b t) q h -> b t q h", b=B)
            atts = einops.rearrange(atts, "(b t) h -> b t h", b=B)
            return inputs, atts

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, rgbs_t, goals_t, actions_t, mask_t, vis_embedded=False):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """
        with self.maybe_autocast():
            B, T = actions_t.shape
            rgbs_t, goals_t, actions_t, mask_t = map(
                lambda t: t.to(self.device), (rgbs_t, goals_t, actions_t, mask_t)
            )

            # if visual inputs have already been embedded through visual encoder, pass through
            if not vis_embedded:
                rgbs_embd, rgbs_attn = self.embed_visual(rgbs_t)
                goals_embd, goals_attn = self.embed_visual(goals_t)
            else:
                rgbs_embd = rgbs_t
                goals_embd = goals_t

            actions_embd = einops.rearrange(
                self.action_embedding(actions_t.long()), "b t h -> b t 1 h"
            )
            mask_t = mask_t.to(torch.bool)

            sa_embds = torch.cat((rgbs_embd, actions_embd), dim=2)
            sa_embds = einops.rearrange(sa_embds, "b t q h -> b (t q) h")
            goals_embd = einops.rearrange(goals_embd, "b t q h -> b (t q) h")
            embd = torch.cat((goals_embd, sa_embds), dim=1)

            import pdb

            pdb.set_trace()
            logits, hx = self.gru(embd)
            logits = logits[:, self.tokens_per_img :: self.tokens_per_img * 2]
            logits = self.action_head(logits)
            probs = F.softmax(logits, dim=-1)

            loss = torch.mean(
                F.cross_entropy(
                    einops.rearrange(probs, "b t h -> (b t) h"),
                    einops.rearrange(actions_t, "b t -> (b t)"),
                    reduction="none",
                )
                * einops.rearrange(mask_t, "b t -> (b t)")
            )

        return NavGRUOutput(loss=loss, probs=probs)

    def action_generator(self, num_envs, deterministic=False, max_its=0):
        """
        action generator function, takes in the next rgb, goal, and action
        """
        T = self.max_trajectory_length

        episodes = [[] for _ in range(num_envs)]

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
                lambda seq: [torch.stack(t, dim=0) for t in seq],
                (rgb_embds, goal_embds),
            )
            actions = [torch.tensor(t) for t in actions]
            rgb_embds, goal_embds, actions_t = map(
                lambda t: self.pad_sequences(t, dim=0).to(self.device),
                (rgb_embds, goal_embds, actions),
            )
            goal_embds = goal_embds[:, 0:1]
            mask_t = torch.ones_like(actions_t, dtype=torch.bool).to(self.device)

            lens = [len(e) for e in episodes]
            max_len = max(lens)

            actions_embd = einops.rearrange(
                self.action_embedding(actions_t.long()), "b t h -> b t 1 h"
            )
            sa_embds = torch.cat((rgb_embds, actions_embd), dim=2)
            sa_embds = einops.rearrange(sa_embds, "b t q h -> b (t q) h")
            goals_embd = einops.rearrange(goal_embds, "b t q h -> b (t q) h")
            embd = torch.cat((goals_embd, sa_embds), dim=1)

            act_pos_delta = [max_len - l + 1 for l in lens]

            logits = self.transformer(embd)[
                :, self.tokens_per_img :: self.tokens_per_img * 2
            ]
            logits = self.action_head(logits)

            probs = F.softmax(logits, dim=-1)

            # project onto the action space
            actions = []

            for i in range(len(episodes)):
                act_probs = probs[i, -act_pos_delta[i]]
                if i == 0:
                    print(act_probs)

                if deterministic:
                    action = act_probs.argmax().cpu().item()
                else:
                    action = torch.multinomial(act_probs, 1).cpu().item()
                actions.append(action)

                episodes[i][-1]["action"] = action

            its += 1

            if its == max_its:
                embd.to("cpu")
                episodes.clear()
                torch.cuda.empty_cache()

            yield actions
