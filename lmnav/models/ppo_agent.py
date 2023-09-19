import torch
from torch import nn

from lmnav.common.utils import logprobs_from_logits

class PPOAgent(nn.Module):

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def action_generator(self, *args):
        return self.actor.action_generator(*args)

    def embed_visual(self, *args):
        return self.actor.embed_visual(*args)

    def forward(self, rgbs_t, goals_t, actions_t, mask_t):
        E, T = rgbs_t.shape[:2]
        device = rgbs_t.device
        
        act_tkn_ids = self.actor.llama_tokenizer('stop forward left right', add_special_tokens=False, return_tensors='pt') 
        act_tkn_ids = act_tkn_ids.input_ids.to(device).squeeze()
        
        output = self.actor.forward_with_embds(rgbs_t, goals_t, actions_t, mask_t)
        logits = output.logits

        act_positions = torch.tensor([(self.actor.tokens_per_img + 1) * (T - i - 1) + 2 for i in range(T)]).to(device)
        actions_t = actions_t.to(torch.int64)

        act_logits = logits[:, -act_positions][:, :, act_tkn_ids]
        values = self.critic(output.hidden_states[-1][:, -act_positions])
        logprobs = logprobs_from_logits(act_logits, actions_t)
        
        return act_logits, values, logprobs
