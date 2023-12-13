from dataclasses import dataclass
import numpy as np
import torch
import inspect

import einops
from torch import nn
import contextlib
import math

from lmnav.models.base_model import BaseModel
from torch.nn import functional as F

@dataclass
class TransformerXLOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    last_hidden_state: torch.Tensor

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(BaseModel):
    def __init__(self, embed_dim, num_heads, use_gtrxl, gtrxl_bias, layer_norm):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Setup GTrXL if used
        self.use_gtrxl = use_gtrxl
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, gtrxl_bias)
            self.gate2 = GRUGate(embed_dim, gtrxl_bias)

        # LayerNorms
        self.layer_norm = layer_norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            h = self.gate1(query, attention)
        else:
            # Skip connection
            h = attention + query
        
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            out = self.gate2(h, forward)
        else:
            # Skip connection
            out = forward + h
        
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)

        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class TransformerXL(BaseModel):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, 
                 vis_encoder,
                 d_hidden,
                 drop_p,
                 n_heads,
                 n_blocks,
                 max_trajectory_length,
                 ln_mode,
                 positional_encoding, 
                 use_gtrxl, 
                 gtrxl_bias,
                 *args,
                 **kwargs) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.
        Arguments:
            input_dim {int} -- Dimension of the input
            max_episode_steps {int} -- Maximum number of steps in an episode
        """
        super().__init__()
        self.num_blocks = n_blocks
        self.embed_dim = d_hidden
        self.num_heads = n_heads
        self.max_trajectory_length = max_trajectory_length
        self.activation = nn.ReLU()
        self.positional_encoding = positional_encoding
        self.use_gtrxl = use_gtrxl
        self.gtrxl_bias = gtrxl_bias
        self.layer_norm = ln_mode
        self.vis_encoder = vis_encoder
        self.max_tokens = (
            max_trajectory_length + (max_trajectory_length + 1) * self.tokens_per_img
        )  # action tokens + rgb tokens + goal token



        # Determine positional encoding
        if positional_encoding == "relative":
            self.wpe = SinusoidalPosition(dim = self.embed_dim)
        elif positional_encoding == "learned":
            self.wpe = nn.Embedding(self.max_tokens, d_hidden)
        else:
            self.wpe = None
        
        # Instantiate transformer blocks
        transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.use_gtrxl, self.gtrxl_bias, self.layer_norm) 
            for _ in range(self.num_blocks)])

        self.transformer = nn.ModuleDict(
            dict(
                action_proj=nn.Embedding(4, self.embed_dim),
                vis_proj = nn.Linear(self.vis_encoder.hidden_size, self.embed_dim),
                drop=nn.Dropout(drop_p),
                h=transformer_blocks,
            )
        )
        self.action_head = nn.Linear(self.embed_dim, 4)

        # Input embedding layer
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

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

    @property
    def tokens_per_img(self):
        return self.vis_encoder.num_tokens

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
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        rgbs_t, goals_t, actions_t, mask_t = map(
            lambda t: t.to(self.device), (rgbs_t, goals_t, actions_t, mask_t)
        )
        actions_t = actions_t.long()
        targets = actions_t.detach().clone().masked_fill_(~mask_t, -100)

        # if visual inputs have already been embedded through visual encoder, pass through
        if not vis_embedded:
            rgbs_embd, goals_embd = self.vis_encoder.embed_obs(rgbs_t, goals_t)
        else:
            rgbs_embd = rgbs_t
            goals_embd = goals_t

        rgbs_embd = rgbs_embd.to(torch.float32)
        goals_embd = goals_embd.to(torch.float32)

        # construct transformer input
        rgbs_embd = self.transformer.vis_proj(rgbs_embd)
        goals_embd = self.transformer.vis_proj(goals_embd)
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
        pos_emb = self.wpe(pos)

        x = self.transformer.drop(embd + pos_emb)
        for block in self.transformer.h:
            block_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=self.device)
            x,_ = block(x, x, x, block_mask)
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

        return TransformerXLOutput(
            loss=loss, logits=act_logits, last_hidden_state=last_hidden_state
        )

           
class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)
