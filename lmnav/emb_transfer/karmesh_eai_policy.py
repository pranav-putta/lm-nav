from typing import Dict, Optional, Tuple

import cv2
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.common import batch_obs
import numpy as np
import torch
from gym import spaces
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ppo import Net, NetPolicy
from torch import nn as nn
from torchvision import transforms as T
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

from lmnav.emb_transfer.sensors import ImageGoalRotationSensor
from lmnav.emb_transfer.transforms import get_transform
from lmnav.emb_transfer.util import load_encoder
from lmnav.emb_transfer.karmesh_visual_encoder import VisualEncoder


class EAINet(Net):
    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space,
            backbone: str,
            resnet_baseplanes: int,
            vit_use_fc_norm: bool,
            vit_global_pool: bool,
            vit_use_cls: bool,
            vit_mask_ratio: float,
            hidden_size: int,
            rnn_type: str,
            num_recurrent_layers: int,
            use_augmentations: bool,
            use_augmentations_test_time: bool,
            randomize_augmentations_over_envs: bool,
            pretrained_encoder: Optional[str],
            freeze_backbone: bool,
            run_type: str,
            avgpooled_image: bool,
            augmentations_name: str,
            drop_path_rate: float,
            scale_obs: float = 1.0
    ):
        super().__init__()

        rnn_input_size = 0
        depth_sensor_uuid = 'depth'
        self.scale_obs = scale_obs

        # visual encoder
        assert "rgb" in observation_space.spaces
        rgb_sensor_shape = observation_space.spaces["rgb"].shape
        rgb_sensor_shape = (
            int(rgb_sensor_shape[0] * scale_obs), int(rgb_sensor_shape[1] * scale_obs), rgb_sensor_shape[2])

        name = "resize"
        if use_augmentations and run_type == "train":
            name = augmentations_name
        if use_augmentations_test_time and run_type == "eval":
            name = augmentations_name
        self.visual_transform = get_transform(name, size=rgb_sensor_shape[0])
        self.visual_transform.randomize_environments = randomize_augmentations_over_envs

        self.visual_encoder = VisualEncoder(
            image_size=rgb_sensor_shape[0],
            backbone=backbone,
            input_channels=3,
            resnet_baseplanes=resnet_baseplanes,
            resnet_ngroups=resnet_baseplanes // 2,
            vit_use_fc_norm=vit_use_fc_norm,
            vit_global_pool=vit_global_pool,
            vit_use_cls=vit_use_cls,
            vit_mask_ratio=vit_mask_ratio,
            avgpooled_image=avgpooled_image,
            drop_path_rate=drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.visual_encoder.output_size, hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        rnn_input_size += 32

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        # pretrained weights
        if pretrained_encoder is not None:
            msg = load_encoder(self.visual_encoder, pretrained_encoder)
            logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))
            msg = load_encoder(self.goal_visual_encoder, pretrained_encoder)
            logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))


        # freeze backbone
        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

                    # save configuration
        self._hidden_size = hidden_size

        self.eval()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def perception_embedding_size(self) -> int:
        return self.visual_encoder.output_size
    
    def recurrent_hidden_size(self) -> int:
        return self._hidden_size

    def forward(
            self,
            observations: Dict[str, torch.Tensor],
            rnn_hidden_states,
            prev_actions,
            masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []

        # number of environments
        N = rnn_hidden_states.size(0)

        # visual encoder
        rgb = observations["rgb"]
        goal = observations["imagegoal"]
        rgb = self.visual_transform(rgb, N)
        goal = self.visual_transform(goal, N)

        embd = self.visual_encoder(rgb, goal)
        embd = self.visual_fc(embd)
        x.append(embd)

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        x.append(prev_actions)

        # state encoder
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)

        return out, rnn_hidden_states, None


@baseline_registry.register_policy
class KarmeshEAIPolicy(NetPolicy):
    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space,
            backbone: str = "resnet18",
            resnet_baseplanes: int = 32,
            vit_use_fc_norm: bool = False,
            vit_global_pool: bool = False,
            vit_use_cls: bool = False,
            vit_mask_ratio: Optional[float] = None,
            hidden_size: int = 512,
            rnn_type: str = "GRU",
            num_recurrent_layers: int = 1,
            use_augmentations: bool = False,
            use_augmentations_test_time: bool = False,
            randomize_augmentations_over_envs: bool = False,
            pretrained_encoder: Optional[str] = None,
            freeze_backbone: bool = False,
            run_type: str = "train",
            avgpooled_image: bool = False,
            augmentations_name: str = "",
            drop_path_rate: float = 0.0,
            **kwargs
    ):
        super().__init__(
            EAINet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                vit_use_fc_norm=vit_use_fc_norm,
                vit_global_pool=vit_global_pool,
                vit_use_cls=vit_use_cls,
                vit_mask_ratio=vit_mask_ratio,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_augmentations=use_augmentations,
                use_augmentations_test_time=use_augmentations_test_time,
                randomize_augmentations_over_envs=randomize_augmentations_over_envs,
                pretrained_encoder=pretrained_encoder,
                freeze_backbone=freeze_backbone,
                run_type=run_type,
                avgpooled_image=avgpooled_image,
                augmentations_name=augmentations_name,
                drop_path_rate=drop_path_rate,
                scale_obs=kwargs['scale_obs'],
            ),
            action_space=action_space,
            # dim_actions=action_space.n,  # for action distribution
        )

    @classmethod
    def from_config(cls,
                    config,
                    observation_space: spaces.Dict,
                    action_space,
                    orig_action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone='vit_small_patch16',
            resnet_baseplanes=32,
            vit_use_fc_norm=False,
            vit_global_pool=False,
            vit_use_cls=False,
            vit_mask_ratio=None,
            hidden_size=512,
            rnn_type='GRU',
            num_recurrent_layers=2,
            use_augmentations=False,
            use_augmentations_test_time=False,
            randomize_augmentations_over_envs=False,
            pretrained_encoder="/srv/flash1/rramrakhya6/summer_2022/mae-for-eai/data/visual_encoders/mae_vit_small_decoder_large_HGPS_RE10K_100.pth",
            freeze_backbone=True,
            run_type='eval',
            avgpooled_image=False,
            augmentations_name='jitter+shift',
            drop_path_rate=0,
            scale_obs=1
        )

    @staticmethod
    def hardcoded(cls, obs_space, action_space):
        return cls(
            observation_space=obs_space,
            action_space=action_space,
            backbone='vit_base_patch16',
            resnet_baseplanes=32,
            vit_use_fc_norm=False,
            vit_global_pool=False,
            vit_use_cls=False,
            vit_mask_ratio=None,
            hidden_size=512,
            rnn_type='LSTM',
            num_recurrent_layers=2,
            use_augmentations=False,
            use_augmentations_test_time=False,
            randomize_augmentations_over_envs=False,
            # pretrained_encoder="/srv/flash1/rramrakhya6/summer_2022/mae-for-eai/data/visual_encoders/mae_vit_small_decoder_large_HGPS_RE10K_100.pth",
            freeze_backbone=True,
            run_type='eval',
            avgpooled_image=False,
            augmentations_name='jitter+shift',
            drop_path_rate=0,
            scale_obs=1
        )

    @staticmethod
    def _construct_state_tensors(num_environments, device):
        rnn_hx = torch.zeros((num_environments, 4, 512), device=device)
        prev_actions = torch.zeros(num_environments, 1, device=device, dtype=torch.long)
        not_done_masks = torch.ones(num_environments, 1, device=device, dtype=torch.bool)

        return rnn_hx, prev_actions, not_done_masks 


    def action_generator(self, num_envs, deterministic):
        rnn_hx, prev_actions, not_done_masks = self._construct_state_tensors(num_envs, self.device)

        while True:
            observations, dones = yield

            for i in range(num_envs):
                if dones[i]:
                    rnn_hx[i] = torch.zeros(rnn_hx.shape[1:])
                    prev_actions[i] = torch.zeros(prev_actions.shape[1:])
                    not_done_masks[i] = torch.ones(not_done_masks.shape[1:])
            
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            features, rnn_hx, _ = self.net(batch, rnn_hx, prev_actions, not_done_masks)
            dist = self.action_distribution(features)

            if deterministic:
                actions = dist.mode()
            else:
                actions = dist.sample()
            
            prev_actions.copy_(actions)

            del observations
            del dones

            yield (actions, dist.probs)        
        
