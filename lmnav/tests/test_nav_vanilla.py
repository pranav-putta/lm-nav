import unittest
from hydra.utils import instantiate

from lmnav.config.default import get_config

import torch


def _init_components(cfg_path, device):
    config = get_config(cfg_path)
    model = instantiate(config.train.policy)

    return model


def construct_dummy_inputs(B, T):
    goals = torch.rand(B, 1, 3, 480, 640)
    rgbs = torch.rand(B, T, 3, 480, 640)
    actions = torch.randint(0, 4, (B, T))

    return goals, rgbs, actions


class TestLoraLLAMATrain(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_load_model(self):
        model = _init_components('train/nav_vanilla/1env/bc/test')


if __name__ == '__main__':
    unittest.main()
