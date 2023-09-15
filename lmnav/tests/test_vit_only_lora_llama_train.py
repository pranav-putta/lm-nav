import unittest
from collections import namedtuple

from hydra.utils import instantiate

from lmnav.common.config import Config
from lmnav.common.registry import registry
from lmnav.config.default import get_config

from lmnav.models import *
from lmnav.processors import *
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from lmnav.common.episode_processor import apply_transforms_inputs

import torch

def _init_components(cfg_path, device):
    config = get_config(cfg_path)
    model = instantiate(config.train.policy)

    agent = model.to(device)
    agent.train()
    
    num_params = sum([param.numel() for param in agent.parameters()])
    num_trainable_params = sum([param.numel() for param in agent.parameters() if param.requires_grad])
    
    print(f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}")
    
    params_with_gradients = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Params with gradients")
    print(params_with_gradients)

    return agent


def construct_dummy_inputs(B, T):
    goals = torch.rand(B, 1, 3, 480, 640)
    rgbs = torch.rand(B, T, 3, 480, 640)
    actions = torch.randint(0, 4, (B, T))

    return goals, rgbs, actions


class TestVITOnly(unittest.TestCase):
    
    def test_vit_only(self):
        cfg_path = "train/nav_llama/lora/1env_clip"
        device = 'cuda'
        B, T = 2, 20

        model = _init_components(cfg_path, device)
        vis_processor = model.vis_encoder.vis_processor
        goals, rgbs, actions = construct_dummy_inputs(B, T)
        rgbs, goals, actions = apply_transforms_inputs(vis_processor, rgbs, goals, actions)

        print("Shapes after transform")
        print(rgbs.shape, goals.shape)
        rgbs = rgbs.to(device)
        goals = goals.to(device) 

        output, atts = model.embed_visual(rgbs)

        print(output.shape)
    
if __name__ == '__main__':
    unittest.main()
