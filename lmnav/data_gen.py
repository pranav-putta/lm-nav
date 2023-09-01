import os

import gym
from habitat_baselines.common import env_spec
import numpy as np

import habitat
import habitat.gym
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
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
from habitat_baselines.utils.info_dict import extract_scalars_from_info


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
    return obs_transforms, env_spec

def _setup_teacher(teacher_ckpt, obs_space, action_space):
    teacher = OldEAIPolicy.hardcoded(OldEAIPolicy, obs_space, action_space)
    torch.set_grad_enabled(False)

    ckpt_dict = torch.load(teacher_ckpt)
    state_dict = ckpt_dict['state_dict']
    state_dict = {k[len('actor_critic.'):]: v for k, v in state_dict.items()}

    teacher.load_state_dict(state_dict)
    return teacher


def _construct_state_tensors(num_environments, device):
    rnn_hx = torch.zeros((num_environments, 2, 512), device=device)
    prev_actions = torch.zeros(num_environments, 1, device=device, dtype=torch.long)
    not_done_masks = torch.ones(num_environments, 1, device=device, dtype=torch.bool)

    return rnn_hx, prev_actions, not_done_masks 
    
    
def collect_episodes(envs, teacher, obs_transform, device,
                     deterministic=False, filter_fn=None, N=None):
    if filter_fn is None:
        filter_fn = lambda _: True
        
    device = torch.device(device)
    num_envs = envs.num_envs
    step = 0
    dataset = []
    episodes = [[] for _ in range(num_envs)]

    rnn_hx, prev_actions, not_done_masks = _construct_state_tensors(num_envs, device)

    teacher.to(device)
    teacher.eval()
    
    observations = envs.reset()

    while (N is None) or (len(dataset) < N):
        # roll out a step
        batch = batch_obs(observations, device)
        batch = apply_obs_transforms_batch(batch, obs_transform)
    
        policy_result = teacher.act(batch,
                                  rnn_hx,
                                  prev_actions,
                                  not_done_masks,
                                  deterministic=deterministic)
        
        prev_actions.copy_(policy_result.actions)
        rnn_hx = policy_result.rnn_hidden_states
    
        step_data = [a.item() for a in policy_result.env_actions.cpu()]
        outputs = envs.step(step_data)
        next_observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
    
        # insert episode into list
        for i, episode in enumerate(episodes):
            episode.append((observations[i], rewards_l[i], infos[i]))
    
        # check if any episodes finished and archive it into dataset
        for i, done in enumerate(dones):
            if done and filter_fn(episodes[i]):
                dataset.append(episodes[i])
                episodes[i] = []
    
                # reset state tensors
                rnn_hx[i] = torch.zeros(rnn_hx.shape[1:])
                prev_actions[i] = torch.zeros(prev_actions.shape[1:])
                not_done_masks[i] = torch.ones(not_done_masks.shape[1:])
    
        observations = next_observations
        step += 1

    return dataset

