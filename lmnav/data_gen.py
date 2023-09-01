import os

import gym
from habitat_baselines.common import env_spec
import numpy as np

import habitat
import habitat.gym
from habitat_baselines.common.env_spec import EnvironmentSpec

from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
import hydra
from habitat_baselines.common.baseline_registry import baseline_registry

import torch

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)

from lmnav.emb_transfer.old_eai_policy import OldEAIPolicy


# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

os.chdir('/srv/flash1/pputta7/projects/lm-nav')

def _init_envs(config=None, is_eval: bool = False):
    env_factory = hydra.utils.instantiate(config.habitat_baselines.vector_env_factory)
    envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )
    _env_spec = EnvironmentSpec(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        orig_action_space=envs.orig_action_spaces[0],
    )

    return envs, _env_spec

def _create_obs_transforms(config, env_spec):
    obs_transforms = get_active_obs_transforms(config)
    env_spec.observation_space = apply_obs_transforms_obs_space(
            env_spec.observation_space, obs_transforms
        )

def _setup_teacher(teacher_ckpt, obs_space, action_space):
    teacher = OldEAIPolicy.hardcoded(OldEAIPolicy, obs_space, action_space)
    torch.set_grad_enabled(False)

    ckpt_dict = torch.load(teacher_ckpt)
    state_dict = ckpt_dict['state_dict']
    state_dict = {k[len('actor_critic.'):]: v for k, v in state_dict.items()}

    teacher.load_state_dict(state_dict)
    return teacher
    

def collect_episodes(envs, teacher, obs_transform, N=None):
    observations = envs.reset()
    episodes = []
    num_envs = len(envs)

    while True:
        batch = batch_obs([observations])
        batch = apply_obs_transforms_batch(batch, obs_transform)
        batch.apply(lambda x: x.to('cuda'))

        

        if N is not None and len(episodes) == N:
            return episodes
            
 
def main():
    config = habitat.get_config("config/imagenav_hm3d.yaml")
    envs, env_spec = _init_envs(config)
    obs_transform = _create_obs_transforms(config, env_spec)
    
    teacher_ckpt = "models/uLHP.300.pth"
    teacher = _setup_teacher(teacher_ckpt, env_spec.observation_space, env_spec.action_space)

    episodes = collect_episodes(envs, teacher, obs_transform)
    

   
if __name__ == "__main__":
    main()
    
