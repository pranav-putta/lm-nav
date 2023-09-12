import os
import multiprocessing as mp
import gc

from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.ddppo.ddp_utils import (
    is_slurm_batch_job,
)
import hydra

import torch


from lmnav.emb_transfer.old_eai_policy import OldEAIPolicy


# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

os.chdir('/srv/flash1/pputta7/projects/lm-nav')



def _init_envs(config=None, is_eval: bool = False):
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


def collect_episodes(config, device, child_conn,
                     deterministic, setup_teacher, filter_fn=None):
    
    print(f"Starting data collection process on device {device}")
    envs, env_spec = _init_envs(config)
    teacher = setup_teacher(config.data_gen.ckpt, config, env_spec)
    
    if filter_fn is None:
        filter_fn = lambda *_: True
        
    device = torch.device(device)
    num_envs = envs.num_envs
    step = 0
    episodes = [[] for _ in range(num_envs)]

    teacher = teacher.to(device)
    teacher = teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    observations = envs.reset()
    dones = [False for _ in range(num_envs)]
    
    total_episodes = 0
    num_succ_episodes = 0

    actor = teacher.action_generator(envs.num_envs, deterministic=deterministic)
    
    while True:
        if child_conn.poll():
            cmd = child_conn.recv()
            if cmd == 'PAUSE':
                print("Puasing dataset collection process...")
                cmd = child_conn.recv()
            
            if cmd == 'EXIT':
                print("Ending dataset collection process...")
                child_conn.close()
                break
            elif cmd == 'RESUME':
                pass
            else:
                raise NotImplementedError(f"Command {cmd} not recognized.")
                
        # roll out a step
        next(actor)
        actions = actor.send((observations, dones))
        step_data = [a.item() for a in actions.cpu()]

        outputs = envs.step(step_data)
        next_observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
        current_episodes = envs.current_episodes()
    
        # insert episode into list
        for i, episode in enumerate(episodes):
            episode.append({'observation': observations[i],
                            'reward': rewards_l[i],
                            'info': infos[i],
                            'action': step_data[i]})

        
        # check if any episodes finished and archive it into dataset
        for i, done in enumerate(dones):
            if done:
                total_episodes += 1
                
                if filter_fn(config, episodes[i]):
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

                    child_conn.send((generator_stats, episode_stats, episodes[i]))
                    
                # reset state tensors
                episodes[i] = []
    
        observations = next_observations
        step += 1

        gc.collect()
        

def filter_fn(config, episode):
    dtg_threshold = config.bc.dtg_threshold
    return len(episode) < 500 and episode[-1]['info']['distance_to_goal'] <= dtg_threshold

def _setup_teacher(teacher_ckpt, config, env_spec):
    obs_space, action_space = env_spec.observation_space, env_spec.action_space
    teacher = OldEAIPolicy.hardcoded(OldEAIPolicy, obs_space, action_space)

    ckpt_dict = torch.load(teacher_ckpt, map_location='cpu')
    state_dict = ckpt_dict['state_dict']
    state_dict = {k[len('actor_critic.'):]: v for k, v in state_dict.items()}

    teacher.load_state_dict(state_dict, strict=False)
    return teacher


def start_data_gen_process(device, config, setup_teacher, filter_fn, deterministic=False):
    """
    This function constructs a multiprocessing server to collect data and returns a queue which can be called to retrieve
    """
    ctx = mp.get_context('forkserver')
    parent_conn, child_conn = ctx.Pipe()

    p = ctx.Process(target=collect_episodes, args=(config, device, 
                                                   child_conn, deterministic, 
                                                   setup_teacher, filter_fn))
    p.start()
    return p, parent_conn 

