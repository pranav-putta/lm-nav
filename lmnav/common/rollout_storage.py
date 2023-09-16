import torch

class RolloutStorage:

    def __init__(self, num_envs, max_steps):
        self.num_envs = num_envs
        self.max_steps = max_steps
        
        self.rgbs = torch.zeros(num_envs, max_steps + 1, 3, 640, 480, dtype=torch.int8) 
        self.goals = torch.zeros(num_envs, max_steps + 1, 3, 640, 480, dtype=torch.int8)
        self.actions = torch.zeros(num_envs, max_steps + 1, dtype=torch.int8)
        self.dones = torch.zeros(num_envs, max_steps + 1,  dtype=torch.bool)
        self.rewards = torch.zeros(num_envs, max_steps + 1, dtype=torch.float16)

        self.current_step_idx = -1
       

    def insert(self, next_observation, dones=None, actions=None, rewards=None):
        """ insert observations, dones, and actions into rollout storage tensors """
        self.rgbs[:, self.current_step_idx + 1] = next_observation['rgb'] 
        self.goals[:, self.current_step_idx + 1] = next_observation['imagegoal']

        if dones is not None:
            self.dones[:, self.current_step_idx] = dones
        if rewards is not None:
            self.rewards[:, self.current_step_idx] = rewards
        if actions is not None:
            self.actions[:, self.current_step_idx] = actions
        
        self.current_step_idx += 1

    def reset(self):
        self.rgbs[:, 0] = self.rgbs[:, -1]
        self.goals[:, 0] = self.goals[:, -1]
        
        self.current_step_idx = 0

    def generate_samples(self):
        """
        TODO; is there a better way to do this??

        1. Take episode sequences with history and apply mask or [sep] tokens
        to fill the space when histories are of different length
        2. Keep one continuous episode accumulation with state history max, and
        use [sep] tokens to differentiate between trajectories

        Again, this assumes that self.max_steps = model max trajectory
        """
        samples = []
        for b in range(self.num_envs):
            ends = torch.where(self.dones[b])[0].tolist()
            if len(ends) >= 1 and ends[0] != 0:
                ends = [0] + ends
            if len(ends) >= 1 and ends[-1] != self.max_steps - 1:
                ends = ends + [self.max_steps - 1]
            slices = [slice(ends[i], ends[i + 1] + 1) for i in range(len(ends) - 1)] 
            
            for s in slices:
                samples.append(tuple(map(lambda t: t[b, s], (self.rgbs, self.goals, self.rewards, self.actions))))
        
        rgbs, goals, actions, rewards = zip(*samples)
        return rgbs, goals, actions, rewards

