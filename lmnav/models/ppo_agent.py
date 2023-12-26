import torch
import einops
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

    def forward(self, rgbs_t, goals_t, prev_actions_t, mask_t, pvk_t, attnm_t):
        # interleaves rgb and action embeddings
        output = self.actor.forward_with_embds(rgbs_t, goals_t, prev_actions_t, mask_t, pvk_t, attnm_t)

        logits = output.logits
        values = self.critic(output.last_hidden_state.to(torch.float32)).squeeze(-1)
        max_episode_len = logits.shape[1]
        logprobs = logprobs_from_logits(logits, prev_actions_t[:, 1:max_episode_len + 1])
        if values.isnan().any():
            print('nan values')
            import pdb; pdb.set_trace()

        return logits, values, logprobs

    def get_values(self, hx):
        return self.critic(hx.to(torch.float32)).squeeze(-1)
