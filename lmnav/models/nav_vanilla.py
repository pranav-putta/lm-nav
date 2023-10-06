from torch import nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


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
    def __init__(self, h_dim, max_T, n_heads, drop_p):
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

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class NavVanillaTransformer(nn.Module):
    def __init__(
            self,
            vis_encoder,
            d_hidden,
            d_head,
            n_heads,
            n_blocks,
            drop_p,
            max_t,
            **kwargs
    ):
        super().__init__()

        self.vis_encoder = vis_encoder
        self.vis_processor = self.vis_encoder.vis_processor

        self.d_hidden = d_hidden
        self.d_head = d_head
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.drop_p = drop_p

        self.transformer = nn.Sequential(
            *[Block(d_hidden, max_t, n_heads, drop_p) for _ in range(n_blocks)]
        )

    @property
    def hidden_size(self):
        return self.d_hidden

    def embed_actions(self, actions_t):
        raise NotImplementedError()

    def embed_visual(self, imgs):
        B, _, T, _, _ = imgs.size()
        image = einops.rearrange(imgs, "b c t h w -> (b t) c h w")
        image_embeds, image_atts = self.vis_encoder.embed_visual(image)

        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        inputs_llama = einops.rearrange(inputs_llama, "(b t) q h -> b t q h", b=B)
        atts_llama = einops.rearrange(atts_llama, "(b t) h -> b t h", b=B)
        return inputs_llama, atts_llama

    def forward(self, rgbs_t, goals_t, actions_t, mask_t):
        """
        rgbs_t = [B, C, T, H, W]
        goals_t = [B, C, 1, H, W]
        actions_t = [B, T]
        """

        rgbs_t, goals_t, mask_t = map(
            lambda t: t.to(self.device), (rgbs_t, goals_t, mask_t)
        )
        rgbs_embd, rgbs_attn = self.embed_visual(rgbs_t)
        goals_embd, goals_attn = self.embed_visual(goals_t)
        mask_t = mask_t.to(torch.bool)

        outputs = self.transformer(embd)

        return outputs
