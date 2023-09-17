from torch import nn

class PPOAgent(nn.Module):

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def action_generator(self, *args):
        return self.actor.action_generator(*args)
    
    def forward(self, rgbs_t, goals_t, actions_t, mask_t):
        output = self.actor(rgbs_t, goals_t, actions_t, mask_t)
        logits = output.logits
        values = self.critic(logits[-1])

        return logits, values
