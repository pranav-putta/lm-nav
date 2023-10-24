from dataclasses import dataclass

from lmnav.common.utils import catchtime, convert_weights_to_fp16, find_tensors
from torch import nn
import inspect

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


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)


        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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
        self.ln_mode = ln_mode

        self.max_trajectory_length = max_trajectory_length
        self.max_tokens = (
            max_trajectory_length + (max_trajectory_length + 1) * self.tokens_per_img
        )  # action tokens + rgb tokens + goal token

        self.transformer = nn.ModuleDict(
            dict(
                action_proj=nn.Embedding(4, self.d_hidden),
                vis_proj=nn.Linear(self.vis_encoder.hidden_size, d_hidden),
                vis_ln=LayerNorm(d_hidden, bias=True),
                wpe=nn.Embedding(self.max_tokens, d_hidden),
                drop=nn.Dropout(drop_p),
                h=nn.ModuleList(
                    [
                        Block(d_hidden, n_heads, True, drop_p, self.max_tokens)
                        for _ in range(n_blocks)
                    ]
                ),
                ln_f=LayerNorm(d_hidden, bias=True),
            )
        )
        self.action_head = nn.Linear(self.d_hidden, 4)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

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
        return rgbs, goals

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer, optim_groups

    def forward_with_embds(self, *args):
        return self.forward(*args, vis_embedded=True)

    def forward(self, rgbs_t, goals_t, actions_t, mask_t, vis_embedded=False):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """

        with self.maybe_autocast():
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

            # construct transformer input
            rgbs_embd = self.transformer.vis_proj(self.transformer.vis_ln(rgbs_embd))
            goals_embd = self.transformer.vis_proj(self.transformer.vis_ln(goals_embd))
            actions_embd = einops.rearrange(
                self.transformer.action_proj(actions_t), "b t h -> b t 1 h"
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
            pos_emb = self.transformer.wpe(pos)

            x = self.transformer.drop(embd + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            last_hidden_state = x[:, self.tokens_per_img :: self.tokens_per_img * 2]

            act_logits = self.action_head(last_hidden_state)
            probs = F.softmax(act_logits, dim=-1)

            print(probs[0])

            loss = F.cross_entropy(
                einops.rearrange(probs, "b t h -> (b t) h"),
                einops.rearrange(targets, "b t -> (b t)"),
                ignore_index=-100,
                label_smoothing=0.1,
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
