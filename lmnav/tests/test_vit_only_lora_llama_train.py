import unittest
from collections import namedtuple

from lmnav.common.config import Config
from lmnav.common.registry import registry
from lmnav.config.default import get_config

from lmnav.models import *
from lmnav.processors import *
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from lmnav.common.episode_processor import apply_transforms_inputs

import torch
import einops

def _init_components(cfg_path, device):
    config = get_config(cfg_path)
    model_config = config.train.policy
    model_cls = registry.get_model_class(model_config._target_)
    model = model_cls.from_config(model_config).to(device)

    vis_processor = registry.get_processor_class(model_config.vis_processor._target_).from_config(model_config.vis_processor)

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


class TestLoraLLAMATrain(unittest.TestCase):
    
    def test_lora_llama_step_loss(self):
        cfg_path = "train/nav_llama/lora/1env_only_vit"
        device = 'cuda'
        B, T = 2, 20

        model, vis_processor = _init_components(cfg_path, device)
        goals, rgbs, actions = construct_dummy_inputs(B, T)
        rgbs, goals, actions = apply_transforms_inputs(vis_processor, rgbs, goals, actions)

        print("Shapes after transform")
        print(rgbs.shape, goals.shape)
        rgbs = rgbs.to(device)
        goals = goals.to(device) 

        output = model.embed_visual(rgbs)
        loss = output.loss.item()

        self.assertTrue(loss >= 0)

    
if __name__ == '__main__':
    unittest.main()
