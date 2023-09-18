import unittest
import torch

from lmnav.common.rollout_storage import RolloutStorage


def construct_dummy_inputs(B):
    goals = torch.rand(B, 32, 4096)
    rgbs = torch.rand(B, 32, 4096)
    actions = torch.randint(0, 4, (B, ))

    return goals, rgbs, actions


class TestRolloutStorage(unittest.TestCase):
    
    def setUp(self):
        self.B = 2
        self.T = 22
        self.max_steps = 22
        self.goals, self.rgbs, self.actions = construct_dummy_inputs(self.B)
        self.rollout_storage = RolloutStorage(self.B, self.max_steps)

        # do initial observation insertion
        self.rollout_storage.insert(self.rgbs, self.goals) 
        
    def do_insertion(self):
        dones = torch.zeros(self.B, dtype=torch.bool)
        rewards = torch.rand(self.B, dtype=torch.float16)
        
        for _ in range(5):
            self.rollout_storage.insert(self.rgbs, self.goals, dones=dones, actions=self.actions, rewards=rewards)

        dones[0] = 1
        self.rollout_storage.insert(self.rgbs, self.goals, dones=dones, actions=self.actions, rewards=rewards)

        dones[0] = 0
        dones[1] = 1
        self.rollout_storage.insert(self.rgbs, self.goals, dones=dones, actions=self.actions, rewards=rewards)

        for _ in range(5):
            self.rollout_storage.insert(self.rgbs, self.goals, dones=dones, actions=self.actions, rewards=rewards)
            
        dones[0] = 1
        self.rollout_storage.insert(self.rgbs, self.goals, dones=dones, actions=self.actions, rewards=rewards)

        
    def test_insertion(self):
        self.do_insertion()

    def test_construct_episodes_tensor(self):
        self.do_insertion()
        samples = self.rollout_storage.generate_samples()
        lengths = [s.shape[0] for s in samples[0]]

        assert lengths[0] == 6
        assert lengths[1] == 7
        assert lengths[2] == 9
        assert lengths[3] == 7

        for i in range(4, 10):
            assert lengths[i] == 1
        assert lengths[-1] == 9
        

if __name__ == "__main__":
    unittest.main()
