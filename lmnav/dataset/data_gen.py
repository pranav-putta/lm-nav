from functools import partial
import pickle
import os
import torch.multiprocessing as mp
import gc
import torch
import copy
import pympler as muppy
import numpy as np

from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.ddppo.ddp_utils import (
    is_slurm_batch_job,
)
import hydra
from pympler import tracker,summary
from pympler import muppy

def _init_envs(config=None, is_eval: bool = False):
    # Quiet the Habitat simulator logging
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    env_factory = hydra.utils.instantiate(config.habitat_baselines.vector_env_factory)
    print(f"Initializing environment on gpu: {config.habitat.simulator.habitat_sim_v0.gpu_device_id}")
    envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
            config.habitat.simulator.habitat_sim_v0.gpu_device_id == 0 
            ),
        )
    _env_spec = EnvironmentSpec(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        orig_action_space=envs.orig_action_spaces[0],
    )

    return envs, _env_spec

def collect_episodes(config, setup_teacher, filter_fn, deterministic, conn, q):
    envs, env_spec = _init_envs(config)
    teacher = setup_teacher(config, env_spec)
    
    if filter_fn is None:
        filter_fn = lambda *_: True
        
    episodes = [[] for _ in range(envs.num_envs)]
    dones = [False for _ in range(envs.num_envs)]
    observations = envs.reset()

    total_episodes = 0
    num_succ_episodes = 0
    step = 0

    actor = teacher.action_generator(envs.num_envs, deterministic=deterministic)

 
    
    while True:
        if conn.poll():
            cmd = conn.recv()
            if cmd == 'PAUSE':
                print("Puasing dataset collection process...")
                cmd = conn.recv()
            
            if cmd == 'EXIT':
                print("Ending dataset collection process...")
                conn.close()
                break
            elif cmd == 'RESUME':
                pass
            else:
                raise NotImplementedError(f"Command {cmd} not recognized.")
                
        # roll out a step
        next(actor)
        actions, probs = actor.send((observations, dones))
        step_data = [a.item() for a in actions.cpu()]

        outputs = envs.step(step_data)
        next_observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
        current_episodes = envs.current_episodes()

        # insert episode into list
        for i, episode in enumerate(episodes):
            episode.append({
               'observation': {k: torch.from_numpy(v).clone() for k, v in observations[i].items()},
                'reward': rewards_l[i],
                'info': {k: torch.from_numpy(v).clone() if isinstance(v, np.ndarray) else v for k, v in infos[i].items()},
                'action': step_data[i],
                'probs': probs[i].cpu()
            })

        
        # check if any episodes finished and archive it into dataset
        for i, done in enumerate(dones):
            if not done:
                continue

            total_episodes += 1
            
            if filter_fn(episodes[i]):
                num_succ_episodes += 1
                
                # log generator stats
                generator_stats = {
                    'generator_step': step,
                    'generator_episode_num': num_succ_episodes,
                    'generator_episode_len': len(episodes[i]),
                    'generator_running_accuracy': num_succ_episodes / total_episodes
                }

                episode_stats = {
                    'scene_id': current_episodes[i].scene_id,
                    'episode_id': current_episodes[i].episode_id
                }

                data = pickle.dumps((generator_stats, episode_stats, episodes[i]))
                q.put(data)

            # reset state tensors
            episodes[i] = []
                
    
        observations = next_observations
        step += 1


def start_data_gen_process(config, setup_teacher, filter_fn, deterministic=False):
    """
    This function constructs a multiprocessing server to collect data and returns a queue which can be called to retrieve

    config: must include config.habitat and config.habitat_baselines for sim
    setup_teacher: function that constructs the actor to be used for generation.
                   the teacher must implement the function action_generator(num_envs, deterministic) which
                   constructs a generator that preserves teacher state
    filter_fn: filter function to decide which episodes to keep
    deterministic: if actions should be sampled from distribution or take argmax
    
    """
    ctx = mp.get_context('forkserver')
    parent_conn, child_conn = ctx.Pipe()
    queue = ctx.Queue(20)

    f = partial(collect_episodes, config=config, setup_teacher=setup_teacher, filter_fn=filter_fn,
                                  deterministic=deterministic, q=queue, conn=child_conn)
    p = ctx.Process(target=f)
    p.start()
    return p, parent_conn, queue
