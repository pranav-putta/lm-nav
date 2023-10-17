from dataclasses import dataclass

from lmnav.common.utils import catchtime, find_tensors
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
class NavVanillaTransformerOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))

        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, ln_mode):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

        self.ln_mode = ln_mode

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        if self.ln_mode == "post":
            x = self.ln1(x + self.attention(x))  # residual
            x = self.ln2(x + self.mlp(x))  # residual
        elif self.ln_mode == "pre":
            x = self.attention(self.ln1(x))
            x = self.mlp(self.ln2(x))
        else:
            raise NotImplementedError(f"layer norm mode {self.ln_mode}")
        return x


class NavVanillaTransformer(BaseModel):
    def __init__(
        self,
        vis_encoder,
        d_hidden,
        d_head,
        n_heads,
        n_blocks,
        drop_p,
        weight_decay,
        max_trajectory_length,
        ln_mode="post",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.vis_encoder = vis_encoder

        self.d_hidden = d_hidden
        self.d_head = d_head
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        self.ln_mode = ln_mode

        self.max_trajectory_length = max_trajectory_length
        self.max_tokens = (
            max_trajectory_length + (max_trajectory_length + 1) * self.tokens_per_img
        )  # action tokens + rgb tokens + goal token

        self.transformer = nn.Sequential(
            *[
                Block(d_hidden, self.max_tokens, n_heads, drop_p, ln_mode)
                for _ in range(n_blocks)
            ]
        )
        self.vis_proj = nn.Linear(self.vis_encoder.hidden_size, self.d_hidden)
        self.action_embedding = nn.Embedding(4, self.d_hidden)
        self.wpe = nn.Embedding(self.max_tokens, self.d_hidden)
        self.action_head = nn.Linear(self.d_hidden, 4)

        self.transformer.apply(self._init_weights)

    @property
    def hidden_size(self):
        return self.d_hidden

    @property
    def tokens_per_img(self):
        return self.vis_encoder.num_tokens

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])

        padded_seqs = [
            F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs
        ]
        return torch.stack(padded_seqs)

    def embed_visual(self, rgbs, goals):
        rgbs, goals = self.vis_encoder.embed_obs(rgbs, goals)
        rgbs = rgbs.float()
        goals = goals.float()
        return rgbs, goals

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optim_groups(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.Conv2d,
            torch.nn.GroupNorm,
        )

        get_optim_params = lambda module: {
            pn: p for pn, p in module.named_parameters() if p.requires_grad
        }

        for mn, m in self.named_modules():
            for pn, p in get_optim_params(m).items():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = get_optim_params(self)
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward_with_embds(self, *args):
        return self.forward(*args, vis_embedded=True)

    def forward(self, rgbs_t, goals_t, actions_t, mask_t, vis_embedded=False):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """
        rgbs_t, goals_t, actions_t, mask_t = map(
            lambda t: t.to(self.device), (rgbs_t, goals_t, actions_t, mask_t)
        )
        actions_t = actions_t.long()
        targets = actions_t.detach().clone().masked_fill_(~mask_t, -100)

        # if visual inputs have already been embedded through visual encoder, pass through
        if not vis_embedded:
            rgbs_embd, goals_embd = self.embed_visual(rgbs_t, goals_t)
        else:
            rgbs_embd = rgbs_t
            goals_embd = goals_t

        rgbs_embd = self.vis_proj(rgbs_embd)
        goals_embd = self.vis_proj(goals_embd)

        actions_embd = einops.rearrange(
            self.action_embedding(actions_t), "b t h -> b t 1 h"
        )
        mask_t = mask_t.to(torch.bool)

        sa_embds = torch.cat((rgbs_embd, actions_embd), dim=2)
        sa_embds = einops.rearrange(sa_embds, "b t q h -> b (t q) h")
        goals_embd = einops.rearrange(goals_embd, "b t q h -> b (t q) h")
        embd = torch.cat((goals_embd, sa_embds), dim=1)

        # add position embeddings
        pos = torch.arange(
            0, embd.shape[1], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        pos_emb = self.wpe(pos)

        logits = self.transformer(embd + pos_emb)
        last_hidden_state = logits[:, self.tokens_per_img :: self.tokens_per_img * 2]

        act_logits = self.action_head(last_hidden_state)
        probs = F.softmax(act_logits, dim=-1)

        loss = F.cross_entropy(
            einops.rearrange(probs, "b t h -> (b t) h"),
            einops.rearrange(targets, "b t -> (b t)"),
            ignore_index=-100,
        )

        return NavVanillaTransformerOutput(
            loss=loss, logits=act_logits, last_hidden_state=last_hidden_state
        )

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
            act_pos_delta = [max_len - l + 1 for l in lens]

            actions = []

            # project onto the action space
            output = self(rgb_embds, goal_embds, actions_t, mask_t, vis_embedded=True)
            probs = F.softmax(output.logits, dim=-1)

            for i in range(len(episodes)):
                act_probs = probs[i, -act_pos_delta[i]]
                if deterministic:
                    action = act_probs.argmax().cpu().item()
                else:
                    action = torch.multinomial(act_probs, 1).cpu().item()
                actions.append(action)

                episodes[i][-1]["action"] = action

            its += 1

            yield actions
