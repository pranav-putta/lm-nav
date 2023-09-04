from lmnav.data_gen import start_data_gen_process
from lmnav.data_gen import _init_envs, _create_obs_transforms, _setup_teacher
from lmnav.common.episode_processor import extract_inputs_from_dataset, sample_subsequences 

import habitat

import time

import unittest
    
from collections import namedtuple

   
class TestEpisodeProcessor(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda:0'
        self.config = habitat.get_config("./lmnav/configs/habitat/imagenav_hm3d.yaml")

    def test_data_gen_process(self):
        B, T = 2, 10
        C, H, W = 3, 480, 640
        process, conn = start_data_gen_process(self.device, self.config, deterministic=False)

        dataset = [conn.recv() for _ in range(1)]
        
        conn.send("EXIT")
        conn.close()
        process.close()
 
        print("Collected episodes!")
        print("Extracting dataset....")

        rgbs, goals, actions = extract_inputs_from_dataset(dataset)
        rgbs_t, goals_t, actions_t = sample_subsequences(2, 10, rgbs, goals, actions) 

        self.assertTrue(rgbs_t.shape == (B, T, C, H, W))
        self.assertTrue(goals_t.shape == (B, 1, C, H, W))
        self.assertTrue(actions_t.shape == (B, T))

        print("Final shape:", rgbs_t.shape)


 
if __name__ == '__main__':
    unittest.main()
