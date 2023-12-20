import torch
from torch import nn

from lmnav.common.utils import logprobs_from_logits


class PPOAgent(nn.Module):
    def __init__(self, actor, critic, **kwargs):
        super().__init__()
        self.actor = actor
        self.critic = critic

    @property
    def vis_processor(self):
        return self.actor.vis_encoder.vis_processor

    def action_generator(self, *args, **kwargs):
        return self.actor.action_generator(*args, **kwargs)

    def embed_visual(self, *args):
        return self.actor.embed_visual(*args)

    def forward(self, rgbs_t, goals_t, actions_t, mask_t, pvk_t, attnm_t):
        actions_t = actions_t.long()
        output = self.actor.forward_with_embds(rgbs_t, goals_t, actions_t, mask_t, pvk_t, attnm_t)

        logits = output.logits
        values = self.critic(output.last_hidden_state.to(torch.float32))
        logprobs = logprobs_from_logits(logits, actions_t)

        return logits, values, logprobs
