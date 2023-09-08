import unittest
from collections import namedtuple

from lmnav.common.config import Config
from lmnav.common.registry import registry

from lmnav.models import *
from lmnav.processors import *
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from lmnav.common.episode_processor import apply_transforms_inputs

import torch
import einops

def _init_components(cfg_path, device):
    Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
    args = Args(cfg_path, "llama_v2", 0, [])

    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    
    model.train()

    vis_processor_cfg = cfg.config.preprocess.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    return model, vis_processor


def construct_dummy_inputs(B, T):
    goals = torch.rand(B, 1, 3, 480, 640)
    rgbs = torch.rand(B, T, 3, 480, 640)
    actions = torch.randint(0, 4, (B, T))

    return goals, rgbs, actions


class TestLoraLLAMATrain(unittest.TestCase):
    
    def test_lora_llama_step_loss(self):
        cfg_path = "/srv/flash1/pputta7/projects/lm-nav/exp_configs/lora_nav_llama_train.yaml"
        device = 'cuda'
        B, T = 2, 20

        model, vis_processor = _init_components(cfg_path, device)
        goals, rgbs, actions = construct_dummy_inputs(B, T)
        rgbs, goals, actions = apply_transforms_inputs(vis_processor, rgbs, goals, actions)

        print("Shapes after transform")
        print(rgbs.shape, goals.shape)
        rgbs = rgbs.to(device)
        goals = goals.to(device) 

        r = 8

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         inference_mode=False,
                         r=r,
                         lora_alpha=32,
                         lora_dropout=0.1)

        model.llama_model = get_peft_model(model.llama_model, peft_config)
        output = model(rgbs, goals, actions)
        loss = output.loss.item()

        self.assertTrue(loss >= 0)

    
if __name__ == '__main__':
    unittest.main()
