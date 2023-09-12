import pickle
import time
import einops
import gc
import glob

from pprint import pprint
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_sim.utils.datasets_download import argparse
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import numpy as np
import random

from collections import namedtuple

import habitat
from habitat import logger
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (init_distrib_slurm, get_distrib_size, rank0_only)
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Any

from torch.utils.data import DataLoader
from lmnav.common.writer import get_writer

from lmnav.data_gen import  _init_envs
from lmnav.common.config import Config as NavLLAMAConfig
from lmnav.models import *
from lmnav.offline_episode_dataset import OfflineEpisodeDataset
from lmnav.processors import *
from lmnav.common.registry import registry as llama_registry
from lmnav.common.episode_processor import apply_transforms_inputs, construct_subsequences, extract_inputs_from_dataset 


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

os.chdir('/srv/flash1/pputta7/projects/lm-nav')


class BCTrainer:
    
    def __init__(self, config, resume_run_id):
        self.config = config
        self.exp_folder = os.path.join(self.config.bc.exp_dir, self.config.bc.exp_name)
        self.resume_run_id = resume_run_id
        
    def initialize_eval(self):
        """
        Initializes controller for evaluation process.
        NOTE: distributed eval is not set up here
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.is_distributed = False
        
        self.writer = get_writer(self.config, self.resume_run_id) 

        self.agent = self.setup_student()
        self.agent.eval()

        
    def initialize_train(self):
        """
        Initializes distributed controller for DDP, starts data generator process
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        local_rank, world_rank, world_size = get_distrib_size()
        self.is_distributed = world_size > 1 or True
        
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)

            # initialize slurm distributed controller
            backend = self.config.habitat_baselines.rl.ddppo.distrib_backend
            print(f"Starting distributed controller using {backend}")
            local_rank, tcp_store = init_distrib_slurm(backend)
            
            self.rank = local_rank

            if rank0_only():
                logger.info(f"Initialized DDP-BC with {torch.distributed.get_world_size()} workers")
                
            # update gpu ids for this process
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank

                self.config.habitat.seed += (torch.distributed.get_rank() * self.config.habitat_baselines.num_environments)

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            
            self.artifact_store = torch.distributed.PrefixStore("artifacts", tcp_store)

           
         
        # set up student
        os.makedirs(self.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, 'ckpts'), exist_ok=True)

        self.agent = self.setup_student()
        
        # set up optimizer
        optim_params = list(filter(lambda p: p.requires_grad, self.agent.parameters()))
        self.optim = torch.optim.Adam(params=optim_params, lr=self.config.bc.lr)

        # set up writer and scatter all relevant data to worker nodes
        if rank0_only(): 
            self.writer = get_writer(self.config, self.resume_run_id) 
            data_files = self.writer.load_dataset(os.path.basename(self.config.bc.data_artifact))
            self.artifact_store.set("data_files", ';'.join(data_files))
        else:
            self.artifact_store.wait(["data_files"])
            data_files = self.artifact_store.get("data_files").decode('utf-8').split(';')


        # set up dataset
        self.dataset = OfflineEpisodeDataset(files=data_files)
        self.data_loader = DataLoader(self.dataset, batch_size=self.config.bc.num_episodes_per_epoch, shuffle=True, collate_fn=lambda x: x, num_workers=4)
        self.data_loader = iter(self.data_loader)

        
                   

    def setup_student(self):
        cfg_path = self.config.bc.llama_cfg
        
        Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
        args = Args(cfg_path, "llama_v2", self.rank, [])

        cfg = NavLLAMAConfig(args)

        model_config = cfg.model_cfg
        model_cls = llama_registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(self.device)

        vis_processor_cfg = cfg.config.preprocess.vis_processor.train
        self.vis_processor = llama_registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

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

    
    def train_epoch(self, epoch):
        num_samples = self.config.bc.batch_size
        max_state_length = self.config.bc.max_trajectory_length
        num_bc_epochs = self.config.bc.bc_epochs
        num_grad_accums = self.config.bc.grad_accums

        episodes = next(self.data_loader)
        rgbs = [einops.rearrange(episode['rgb'], 't h w c -> t c h w') for episode in episodes]
        goals = [einops.rearrange(episode['imagegoal'], 't h w c -> t c h w') for episode in episodes]
        actions = [episode['action'] for episode in episodes]

        total_loss = 0
        total_samples = num_samples * num_bc_epochs * num_grad_accums
        rgbs, goals, actions = construct_subsequences(total_samples, max_state_length, rgbs, goals, actions)
        p_idxs = torch.randperm(len(rgbs)).view(num_bc_epochs, num_grad_accums, num_samples)
       
        for bc_epoch in range(num_bc_epochs):
            for i in range(num_grad_accums):
                idxs = p_idxs[bc_epoch, i] 
                # construct batch
                rgbs_t, goals_t, actions_t = map(lambda t: [t[i] for i in idxs], (rgbs, goals, actions))
                T = max_state_length # TODO update this to max length, but for testing keep at T
                # pad inputs to T
                mask_t = torch.stack([torch.cat([torch.ones(t.shape[0]), torch.zeros(T - t.shape[0])]) for t in rgbs_t])
                mask_t = mask_t.bool()
                rgbs_t = torch.stack([F.pad(t, (0,)*7 + (T - t.shape[0],), 'constant', 0) for t in rgbs_t])
                goals_t = torch.stack(goals_t) 
                actions_t = torch.stack([F.pad(t, (0, T - t.shape[0]), 'constant', 0) for t in actions_t])
                rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)
                
                if i < num_grad_accums - 1:
                    with self.agent.no_sync():
                        outputs = self.agent(rgbs_t, goals_t, actions_t, mask_t)
                        loss = outputs.loss
                        total_loss += loss.item()
                        loss.backward()
                else:
                    outputs = self.agent(rgbs_t, goals_t, actions_t, mask_t)
                    loss = outputs.loss
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
            
                rgbs_t.to('cpu')
                goals_t.to('cpu')

        del rgbs
        del goals
        del actions
        gc.collect()
            
        avg_loss = total_loss / (num_bc_epochs * num_grad_accums)

        return {
            'loss': avg_loss,
            'epoch': epoch
        }
        
        
    def _all_reduce(self, t):
        if not self.is_distributed:
            return t

        orig_device = t.device
        t = t.to(self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)
        

    def save_checkpoint(self, epoch):
        # only save parameters that have been updated
        param_grad_dict = {
            k: v.requires_grad for k, v in self.agent.named_parameters()
        }

        state_dict = self.agent.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dict.keys() and not param_grad_dict[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optim.state_dict(),
            "config": OmegaConf.to_container(self.config),
            "epoch": epoch
        }
        ckpt_num = epoch // self.config.bc.ckpt_freq
        ckpt_filepath = os.path.join(self.exp_folder, 'ckpts', f'ckpt.{ckpt_num}.pth')
        torch.save(save_obj, ckpt_filepath) 

        self.writer.save_artifact(self.config.bc.exp_name, 'model', os.path.abspath(ckpt_filepath))

        
    def train(self):
        self.initialize_train()
        

        epochs, batch_size = self.config.bc.epochs, self.config.bc.batch_size

        for epoch in range(epochs):
            stats = self.train_epoch(epoch)
            torch.distributed.barrier()
            stats_keys = sorted(stats.keys())
            stats = torch.tensor([stats[key] for key in stats_keys],
                                 device='cpu',
                                 dtype=torch.float32)
            stats = self._all_reduce(stats)
            stats /= torch.distributed.get_world_size()

            stats = { k: stats[i].item() for i, k in enumerate(stats_keys) }

            if rank0_only():
                self.writer.write(stats)

                if epoch % self.config.bc.ckpt_freq == 0:
                    self.save_checkpoint(epoch)
                   

                    
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
        eval_folder = os.path.join(self.exp_folder, 'eval')
        ckpt_path_pattern = os.path.join(os.path.join(self.exp_folder, 'ckpts'), self.config.bc.eval.ckpt)
        ckpt_paths = glob.glob(ckpt_path_pattern)

        # go through each ckpt and get previous stats
        for i in range(len(ckpt_paths)):
            stats_path = os.path.join(eval_folder, os.path.basename(ckpt_paths[i]), 'stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    prev_stats = pickle.load(f)
                ckpt_paths[i] = (ckpt_paths[i], prev_stats)  
            else:
                ckpt_paths[i] = (ckpt_paths[i], None)
        
        ckpt_paths = reversed(sorted(list(ckpt_paths), key=lambda x: int(x[0].split(".")[1])))
        envs, env_spec = _init_envs(self.config)
        self.initialize_eval()

        for ckpt_path, stats in ckpt_paths:
            self.eval_checkpoint(ckpt_path, stats, envs)
        
        
    def eval_checkpoint(self, ckpt_path, prev_stats, envs):
        print(f"Starting evaluation for {ckpt_path}")
        
        N_episodes = self.config.bc.eval.num_episodes
        T = self.config.bc.max_trajectory_length

        # construct directory to save stats
        ckpt_name = os.path.basename(ckpt_path)
        eval_dir = os.path.join(self.exp_folder, 'eval', ckpt_name)
        video_dir = os.path.join(eval_dir, 'videos')
        os.makedirs(eval_dir, exist_ok=True)
        
        if self.config.bc.eval.save_videos:
            os.makedirs(video_dir, exist_ok=True)

        # load checkpoint
        print(f"Loading model from checkpoint")
        ckpt_state_dict = torch.load(ckpt_path)
        ckpt_state_dict = { k[len('module.'):]:v for k, v in ckpt_state_dict['model'].items() }
        self.agent.load_state_dict(ckpt_state_dict, strict=False)

        # turn of all gradients
        for param in self.agent.parameters():
            param.requires_grad = False

        observations = envs.reset()
        episodes = [[] for _ in range(envs.num_envs)]
        episode_idxs_to_reset = set()

        stats = {
            f'{ckpt_name}/total_episodes': 0,
            f'{ckpt_name}/successful_episodes': 0,
        }

        if prev_stats is not None:
            stats = prev_stats

        actor = self.agent.action_generator(envs.num_envs, T, self.vis_processor, deterministic=False)
        
        while stats[f'{ckpt_name}/total_episodes'] < N_episodes:
            
            next(actor)
            actions = actor.send((observations, episode_idxs_to_reset)) 
            episode_idxs_to_reset = set()
                        
            outputs = envs.step(actions)
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
                if done:
                    stats[f'{ckpt_name}/total_episodes'] += 1
                    
                    if episodes[i][-1]['info']['distance_to_goal'] < self.config.bc.dtg_threshold:
                        stats[f'{ckpt_name}/successful_episodes'] += 1

                    self.writer.write(stats)
                    if self.config.bc.eval.save_videos:
                        try:
                            ckpt_idx = ckpt_name.split('.')[1]
                            self.save_episode_video(episodes[i], stats[f'{ckpt_name}/total_episodes'], video_dir, ckpt_idx)
                        except:
                            print("There was an error while saving video!")

                    # this is to tell actor generator to clear this episode from history
                    episode_idxs_to_reset.add(i)
                    episodes[i] = []


            observations = next_observations
        
            with open(os.path.join(eval_dir, 'stats.pkl'), 'wb+') as f:
                pickle.dump(stats, f)
         
               


def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    parser.add_argument('--eval', action='store_true', help='Flag to enable evaluation mode')
    parser.add_argument('--resume_run_id', type=str, help="Writer run id to restart")
    args = parser.parse_args()

    config = habitat.get_config(args.cfg_path)
    trainer = BCTrainer(config, args.resume_run_id)

    if not args.eval:
        trainer.train()
    else:
        with read_write(config):
            config.bc.mode = 'eval'
            config.habitat_baselines.wb.group = 'eval'
            config.habitat_baselines.wb.run_name = f'eval {config.habitat_baselines.wb.run_name}'
            # config.habitat.dataset.split = 'val_hard'
     
        trainer.eval()


if __name__ == "__main__":
    main()
    
    
    
