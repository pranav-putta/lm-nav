import unittest
from hydra.utils import instantiate

from lmnav.config.default import get_config
from lmnav.common.episode_processor import apply_transforms_inputs

import torch
import pdb


def _init_components(cfg_path, device):
    config = get_config(cfg_path)
    model = instantiate(config.train.policy)
    model = model.to(device)
    return model


def construct_dummy_inputs(B, T):
    goals = torch.rand(B, 1, 3, 480, 640)
    rgbs = torch.rand(B, T, 3, 480, 640)
    actions = torch.randint(0, 4, (B, T))
    masks = torch.randint(0, 2, (B, T)).bool()

    return rgbs, goals, actions, masks


class TestLoraLLAMATrain(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_model_forward(self):
        model = _init_components("train/nav_vanilla/1env/bc/test", "cuda")
        rgbs, goals, actions, masks = construct_dummy_inputs(2, 100)
        rgbs, goals, actions = apply_transforms_inputs(
            model.vis_processor, rgbs, goals, actions
        )

        out = model(rgbs, goals, actions, masks)
        print(sum([p.numel() for p in model.parameters() if p.requires_grad]))


if __name__ == "__main__":
    unittest.main()
