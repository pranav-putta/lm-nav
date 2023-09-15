import unittest
import torch

from lmnav.common.rollout_storage import RolloutStorage


def construct_dummy_inputs(B):
    goals = torch.rand(B, 3, 224, 224)
    rgbs = torch.rand(B, 3, 224, 224)
    actions = torch.randint(0, 4, (B, ))

    return goals, rgbs, actions


class TestRolloutStorage(unittest.TestCase):
    
    def setUp(self):
        self.B = 2
        self.T = 22
        self.max_steps = 22
        self.goals, self.rgbs, self.actions = construct_dummy_inputs(self.B)
        self.rollout_storage = RolloutStorage(self.B, self.max_steps, 224)

        # do initial observation insertion
        self.rollout_storage.insert({'rgb': self.rgbs, 'imagegoal': self.goals}) 
        
    def do_insertion(self):
        dones = torch.zeros(self.B, dtype=torch.bool)
        rewards = torch.rand(self.B, dtype=torch.float16)
        obs = {'rgb': self.rgbs, 'imagegoal': self.goals}
        
        for _ in range(5):
            self.rollout_storage.insert(obs, dones=dones, actions=self.actions, rewards=rewards)

        dones[0] = 1
        self.rollout_storage.insert(obs, dones=dones, actions=self.actions, rewards=rewards)

        dones[0] = 0
        dones[1] = 1
        self.rollout_storage.insert(obs, dones=dones, actions=self.actions, rewards=rewards)

        for _ in range(5):
            self.rollout_storage.insert(obs, dones=dones, actions=self.actions, rewards=rewards)
            
        dones[0] = 1
        self.rollout_storage.insert(obs, dones=dones, actions=self.actions, rewards=rewards)

        
    def test_insertion(self):
        self.do_insertion()

    def test_construct_episodes_tensor(self):
        self.do_insertion()
        self.rollout_storage._construct_episodes_tensor()
        

if __name__ == "__main__":
    unittest.main()
