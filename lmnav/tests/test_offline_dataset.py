from lmnav.data_gen import start_data_gen_process
from lmnav.data_gen import _init_envs, _create_obs_transforms, _setup_teacher
from lmnav.common.episode_processor import extract_inputs_from_dataset, sample_subsequences 

import habitat

import time

import unittest
    
from collections import namedtuple
from lmnav.datasets import OfflineEpisodeDataset

   
class TestOfflineDataset(unittest.TestCase):
    
    def test_get_item_offline_dataset(self):
        dataset = OfflineEpisodeDataset('./offline/output')
        
        for i in range(100):
            x = dataset[i]
            rgb = x['rgb']
            goal = x['imagegoal']
            self.assertTrue(tuple(rgb.shape[1:]) == (480, 640, 3))
            self.assertTrue(tuple(goal.shape) == (1, 480, 640, 3))
            print(f"Verified episode {i}")

 
if __name__ == '__main__':
    unittest.main()
