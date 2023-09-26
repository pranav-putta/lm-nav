import pickle
import re 

import einops

from pprint import pprint
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_sim.utils.datasets_download import argparse
from habitat_baselines.utils.info_dict import extract_scalars_from_info

from hydra.utils import instantiate

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (rank0_only)

import torch
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from lmnav.config.default import get_config

from lmnav.dataset.data_gen import  _init_envs
from lmnav.models import *
from lmnav.models.base_policy import instantiate_model
from lmnav.processors import *
from lmnav.common.episode_processor import apply_transforms_images 

from lmnav.common.writer import *
from lmnav.dataset.offline_episode_dataset import *


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

class EvalRunner:
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.exp_folder = os.path.join(self.config.exp.root_dir,
                                       self.config.exp.group,
                                       self.config.exp.job_type,
                                       self.config.exp.name)
        self.writer = instantiate(self.config.exp.logger, eval_mode=True)

    def current_episodes(self):
        for write_fn in self.envs._connection_write_fns:
            write_fn(("call", ("current_episode", {'all_info': True})))
        results = []
        for read_fn in self.envs._connection_read_fns:
            results.append(read_fn())
        return results
    
    def auto_find_resume_run(self):
        runs = self.writer.load_runs(filters={"config.exp.name": self.config.exp.name,
                                       "config.exp.group": self.config.exp.group,
                                       "config.exp.job_type": self.config.exp.job_type})                                    
        # filter for eval runs
        runs = list(filter(lambda r: r.job_type == 'eval', runs))
        assert len(runs) <= 1, "found multiple eval jobs for this run. not sure what to do"

        if len(runs) == 1:
            with read_write(self.config):
                self.config.exp.resume_id = runs[0].id
        
    def initialize_eval(self):
        """
        Initializes controller for evaluation process.
        NOTE: distributed eval is not set up here
        """
        self.validate_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.is_distributed = False
        self.eval_dir = os.path.join(self.exp_folder, 'eval')

        self.auto_find_resume_run()
        self.writer.open(self.config,
                         override_run_name=f'eval {os.path.join(self.config.exp.job_type,self.config.exp.name)}')
        
        self.envs, env_spec = _init_envs(self.config)
        self.agent = self.setup_student()
        self.agent.eval()

        self.tablecols = ['ckpt', 'episode_id', 'difficulty', 'geodesic_distance', 'success']
        self.stats_path = os.path.join(self.eval_dir, 'stats.pkl')
        
        self.table_data = []
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'rb') as f:
                self.table_data = pickle.load(f)
        
        
    def validate_config(self):
        pass
        
    def setup_student(self):
        OmegaConf.resolve(self.config)
        model = instantiate_model(self.config.eval.policy, writer=self.writer, store=None)

        self.vis_processor = model.vis_processor

        agent = model.to(self.device)
        agent.train()

        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            agent = DDP(agent, device_ids=[self.rank])

        num_params = sum([param.numel() for param in agent.parameters()])
        num_trainable_params = sum([param.numel() for param in agent.parameters() if param.requires_grad])

        print(f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}")

        params_with_gradients = [name for name, param in model.named_parameters() if param.requires_grad]
        if rank0_only():
            print("Params with gradients")
            pprint(params_with_gradients)

        return agent


    def load_checkpoint(self, ckpt_path):
        print(f"Loading model from checkpoint: {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path)
        ckpt_state_dict = { k[len('module.'):]:v for k, v in ckpt_state_dict['model'].items() }
        self.agent.load_state_dict(ckpt_state_dict, strict=False)
        self.agent.eval()

        
    def save_episode_video(self, episode, num_episodes, video_dir, ckpt_idx):
        obs_infos = [(step['observation'], step['info']) for step in episode]
        _, infos = zip(*obs_infos)

        frames = [observations_to_image(obs, info) for obs, info in obs_infos]
        disp_info = {k: [info[k] for info in infos] for k in infos[0].keys()}

        generate_video(
            video_option=['disk'],
            video_dir=video_dir,
            images=frames,
            episode_id=num_episodes,
            checkpoint_idx=ckpt_idx,
            metrics=extract_scalars_from_info(disp_info),
            fps=self.config.habitat_baselines.video_fps,
            tb_writer=None,
            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name
        )


    def eval(self):
        self.initialize_eval()

        if self.config.eval.policy.load_artifact.version == '*':
            versions = self.writer.load_model_versions(self.config.eval.policy.load_artifact)
            versions = reversed(sorted(versions, key=lambda t: int(t[1:])))  
        else:
            versions = [self.config.eval.policy.load_artifact.version]
            
        for version in versions:
            with read_write(self.config):
                self.config.eval.policy.load_artifact.version = version
            ckpt_path = self.writer.load_model(self.config.eval.policy.load_artifact)
            self.eval_checkpoint(ckpt_path)


    def embed_observations(self, observations):
        observations = batch_obs(observations, self.device)
        rgbs, goals = map(lambda t: einops.rearrange(t, 'b h w c -> b 1 c h w'), (observations['rgb'], observations['imagegoal']))
        rgbs_t, goals_t = apply_transforms_images(self.vis_processor, rgbs, goals) 
        img_embds_t, img_atts_t = self.agent.embed_visual(torch.cat([rgbs_t, goals_t], dim=2).to(self.device))
        rgb_embds, goal_embds = img_embds_t[:, 0], img_embds_t[:, 1]

        map(lambda t: t.to('cpu'), (observations['rgb'], observations['imagegoal'], observations['depth']))
        del observations
        return rgb_embds, goal_embds


    def eval_checkpoint(self, ckpt_path):
        print(f"Starting evaluation for {ckpt_path}")

        N_episodes = self.config.eval.num_episodes

        # construct directory to save stats
        ckpt_name = os.path.basename(ckpt_path)
        eval_dir = os.path.join(self.eval_dir, ckpt_name)
        video_dir = os.path.join(eval_dir, 'videos')
        os.makedirs(eval_dir, exist_ok=True)

        if self.config.eval.save_videos:
            os.makedirs(video_dir, exist_ok=True)

        # load checkpoint
        self.load_checkpoint(ckpt_path)

        # turn of all gradients
        for param in self.agent.parameters():
            param.requires_grad = False

        observations = self.envs.reset()
        episodes = [[] for _ in range(self.envs.num_envs)]
        dones = [False for _ in range(self.envs.num_envs)]
        episode_infos = self.current_episodes()
        
        num_episodes_done = len(list(filter(lambda r: r[0] == ckpt_name, self.table_data)))
        num_success = 0

        actor = self.agent.action_generator(self.envs.num_envs, deterministic=self.config.eval.deterministic)

        while num_episodes_done < N_episodes:
            next(actor)
            actions = actor.send((self.embed_observations(observations), dones)) 

            outputs = self.envs.step(actions)
            next_observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)] 

            # add environment observation to episodes list
            for i in range(len(episodes)):
                episodes[i].append({
                    'observation': observations[i],
                    'reward': rewards_l[i],
                    'info': infos[i],
                    'action': actions[i]
                })

            for i, done in enumerate(dones):
                if not done:
                    continue
                
                num_episodes_done += 1
                success = episodes[i][-1]['info']['distance_to_goal'] < self.config.eval.dtg_threshold
                num_success += int(success)

                stats = {"ckpt": ckpt_name,
                         "episode_id": episode_infos[i].episode_id,
                         "difficulty": episode_infos[i].info['difficulty'],
                         "geodesic_distance": episode_infos[i].info["geodesic_distance"],
                         "success": success}
                
                self.table_data.append([stats[k] for k in self.tablecols])

                self.writer.write({'table': wandb.Table(columns=self.tablecols, data=self.table_data),
                                  'ckpt': ckpt_name,
                                  'success_rate': (num_success / num_episodes_done),
                                   f'{ckpt_name}/successful_episodes': num_success})
                with open(self.stats_path, 'wb+') as f:
                    pickle.dump(self.table_data, f)
                if self.config.eval.save_videos:
                    try:
                        ckpt_idx = ckpt_name.split('.')[1]
                        self.save_episode_video(episodes[i], num_episodes_done, video_dir, ckpt_idx)
                    except:
                        print("There was an error while saving video!")

                # this is to tell actor generator to clear this episode from history
                episodes[i] = []
            
            episode_infos = self.current_episodes()
            observations = next_observations
            
def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    parser.add_argument('--debug', action='store_true', help='Flag to enable debug mode')
    parser.add_argument('--deterministic', action='store_true', help='Flag to quickly enable determinism')
    parser.add_argument('--version', type=str, help='Which version of the model to run')
    args = parser.parse_args()

    config = get_config(args.cfg_path)

    with read_write(config):
        config.habitat_baselines.num_environments = config.eval.num_envs
        config.eval.deterministic = args.deterministic
        if args.version:
            config.eval.policy.load_artifact.version = args.version

    runner = EvalRunner(config, verbose=args.debug)
    runner.eval()


if __name__ == "__main__":
    main()
    
