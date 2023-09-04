from lmnav.common.registry import registry
from lmnav.common.config import Config

import torch
import unittest
import random

from lmnav.common.episode_processor import apply_transforms_inputs, sample_subsequences
    
from collections import namedtuple

def _init_vis_processor(cfg_path):
    Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
    args = Args(cfg_path, "llama_v2", 0, [])

    cfg = Config(args)
    
    vis_processor_cfg = cfg.config.preprocess.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    return vis_processor

    
class TestEpisodeProcessor(unittest.TestCase):
    
    def setUp(self):
        cfg_path = "/srv/flash1/pputta7/projects/lm-nav/exp_configs/lin_nav_llama_train.yaml"
        self.vis_processor = _init_vis_processor(cfg_path)
        
    def construct_inputs(self, B, T):
        goals = torch.rand(B, 1, 3, 480, 640)
        rgbs = torch.rand(B, T, 3, 480, 640)
        actions = torch.randint(0, 4, (B, T)).tolist()

        return goals, rgbs, actions
 
    def test_transform_shapes(self):
        B, T = 10, 20

        rgbs, goals, actions = self.construct_inputs(B, T) 
        rgbs, goals, actions = apply_transforms_inputs(self.vis_processor, rgbs, goals, actions)

        self.assertTrue(tuple(rgbs.shape) == (B, 3, T, 224, 224))
        self.assertTrue(tuple(goals.shape) == (B, 3, 1, 224, 224))

    def test_sample_subsequences(self):
        B, T = 10, 20
        C, H, W = 3, 480, 640
        ts = [random.randint(0, 500) for _ in range(B)]
        rgbs = [torch.rand(ts[i], 3, 480, 640) for i in range(B)]
        goals = [torch.rand(1, 3, 480, 640) for _ in range(B)]
        actions = [torch.rand(ts[i]) for i in range(B)]

        rgbs_t, goals_t, actions_t = sample_subsequences(B, T, rgbs, goals, actions) 

        print(rgbs_t.shape, goals_t.shape, actions_t.shape)
        self.assertTrue(tuple(rgbs_t.shape) == (B, T, C, H, W))
        self.assertTrue(tuple(goals_t.shape) == (B, T, C, H, W))


if __name__ == '__main__':
    unittest.main()
