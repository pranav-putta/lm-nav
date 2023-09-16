from torch import nn

class PPOAgent(nn.Module):

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def action_generator(self, *args):
        return self.actor.action_generator(*args)
