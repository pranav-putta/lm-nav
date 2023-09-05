import time

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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import multiprocessing as mp

from typing import Any
from lmnav.common.writer import get_writer

from lmnav.data_gen import start_data_gen_process, _init_envs
from lmnav.common.config import Config as NavLLAMAConfig
from lmnav.models import *
from lmnav.processors import *
from lmnav.common.registry import registry as llama_registry
from lmnav.common.episode_processor import apply_transforms_inputs, extract_inputs_from_dataset, sample_subsequences


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

os.chdir('/srv/flash1/pputta7/projects/lm-nav')



class BCTrainer:
    data_process: Any 
    data_conn: Any

    
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.config = habitat.get_config(self.cfg_path)

        
        
    def initialize_eval(self):
        """
        Initializes controller for evaluation process.
        NOTE: distributed eval is not set up here
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.is_distributed = False
        
        self.writer = get_writer(self.config) 

        self.agent = self.setup_student()
        self.agent.eval()

        
    def initialize_train(self):
        """
        Initializes distributed controller for DDP, starts data generator process
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        local_rank, world_rank, world_size = get_distrib_size()
        self.is_distributed = world_size > 1
        
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
            
            self.num_rollouts_done_store = torch.distributed.PrefixStore("rollout_tracker", tcp_store)
            self.num_rollouts_done_store.set("num_done", "0")

            
        # start data generator process
        print(f"Now starting gen process on {self.rank}")
        self.data_process, self.data_conn = start_data_gen_process(self.device, self.config, deterministic=False)
        
      
        # set up student
        self.agent = self.setup_student()
        os.makedirs(self.config.bc.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.config.bc.exp_folder, 'ckpts'), exist_ok=True)

        # set up optimizer
        optim_params = list(filter(lambda p: p.requires_grad, self.agent.parameters()))
        self.optim = torch.optim.Adam(params=optim_params, lr=self.config.bc.lr)
        self.dataset = []
        
        if rank0_only(): 
            self.writer = get_writer(self.config) 

            

    def setup_student(self):
        cfg_path = "/srv/flash1/pputta7/projects/lm-nav/exp_configs/lin_nav_llama_train.yaml"
        
        Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
        args = Args(cfg_path, "llama_v2", self.rank, [])

        cfg = NavLLAMAConfig(args)

        model_config = cfg.model_cfg
        model_cls = llama_registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(self.device)
        model.train()

        vis_processor_cfg = cfg.config.preprocess.vis_processor.train
        self.vis_processor = llama_registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        agent = model.to(self.device)
        
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
        num_samples = 2
        max_state_length = 20
        num_bc_epochs = 10
        min_episodes = 5
        max_episodes = 20
        num_grad_accums = 6

        # accumulate data from the data queue; ideally the queue should already be filled
        start_episode_wait = time.time()
        episode_stats, episodes = zip(*[self.data_conn.recv() for _ in range(min_episodes)])
        avg_generator_stats = { k: sum([stats[k] for stats in episode_stats]) / len(episode_stats) for k in episode_stats[0].keys() } 
        self.dataset += episodes        
        end_episode_wait = time.time()
        
        if len(self.dataset) > max_episodes:
            self.dataset = self.dataset[len(self.dataset) - max_episodes:]

        rgbs, goals, actions = extract_inputs_from_dataset(self.dataset)

        total_loss = 0

        for bc_epoch in range(num_bc_epochs):
            for i in range(num_grad_accums):
                rgbs_t, goals_t, actions_t = sample_subsequences(num_samples, max_state_length, rgbs, goals, actions)
                rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)
                
                if i < num_grad_accums - 1:
                    with self.agent.no_sync():
                        outputs = self.agent(rgbs_t, goals_t, actions_t)
                        loss = outputs.loss
                        total_loss += loss.item()
                        loss.backward()
                else:
                    outputs = self.agent(rgbs_t, goals_t, actions_t)
                    loss = outputs.loss
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
            
                rgbs_t.to('cpu')
                goals_t.to('cpu')
            
        avg_loss = total_loss / (num_bc_epochs * num_grad_accums)

        return {
            'loss': avg_loss,
            'epoch': epoch,
            'episode_wait': (end_episode_wait - start_episode_wait),
            **avg_generator_stats
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
        ckpt_folder = os.path.join(self.config.bc.exp_folder, 'ckpt')
        torch.save(save_obj,  os.path.join(ckpt_folder, f'ckpt.{ckpt_num}.pth')) 

        
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
                   

                    
    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])
        
        padded_seqs = [F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs]
        return torch.stack(padded_seqs)

    def eval_checkpoint(self, ckpt_path):
        N_episodes = 100
        T = 20
        
        # start evaluation
        print(f"Starting evaluation of checkpoint {ckpt_path}")
        num_episodes_done = 0
        envs, env_spec = _init_envs(self.config)
        self.initialize_eval()

        # load checkpoint
        print(f"Loading model from checkpoint")
        ckpt_state_dict = torch.load(os.path.join(self.config.bc.exp_folder, ckpt_path))
        ckpt_state_dict = { k[len('module.'):]:v for k, v in ckpt_state_dict.items() }
        self.agent.load_state_dict(ckpt_state_dict)

        # turn of all gradients
        for param in self.agent.parameters():
            param.requires_grad = False

        past_kv = [None for _ in range(envs.num_envs)]
        observations = envs.reset()

        episodes = [[] for _ in range(envs.num_envs)]

        stats = {
            'total_episodes': 0,
            'successful_episodes': 0 
        }

        
        tokens = self.agent.llama_tokenizer('stop forward left right', add_special_tokens=False, return_tensors='pt') 
        tokens = tokens.input_ids.to(self.device).squeeze()

        
        while num_episodes_done < N_episodes:
            batch = batch_obs(observations, self.device)

            for i in range(envs.num_envs):
                episodes[i].append({
                    'observation': observations[i],
                    'action': 0
                })
                
            partial_episodes = [e[-(T - 1):] for e in episodes][:2]

            rgbs, goals, actions = extract_inputs_from_dataset(partial_episodes) 
            rgbs_t = self.pad_sequences(rgbs, dim=0)
            actions_t = self.pad_sequences(actions, dim=0)
            goals_t = self.pad_sequences(goals, dim=0)
            lens = [len(e) for e in partial_episodes]
            max_len = max(lens)

            rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)
            outputs = self.agent(rgbs_t, goals_t, actions_t)

            act_pos_delta = [33 * (max_len - l) + 2 for l in lens]
            logits = outputs.logits


            # project onto the action space
            actions = []
            for i in range(len(partial_episodes)):
                act_logits = logits[i, -act_pos_delta[i], tokens]
                act_logits = F.softmax(act_logits)
                actions.append(act_logits.argmax().cpu().item())
                
            outputs = envs.step(actions)
            next_observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)] 

            for i in range(len(episodes)):
                episodes[i][-1] = {
                    'observation': observations[i],
                    'reward': rewards_l[i],
                    'info': infos[i],
                    'action': actions[i]
                }
            
            for i, done in enumerate(dones):
                if done:
                    stats['total_episodes'] += 1
                    
                    if episodes[i][-1]['info']['distance_to_goal'] < self.config.bc.dtg_threshold:
                        stats['successful_episodes'] += 1

                    self.writer.write(stats)

                    if self.config.bc.eval.save_videos:
                        vid_folder = os.path.join(self.config.bc.exp_folder, 'videos')
                        os.makedirs(vid_folder, exist_ok=True)
                        
                        obs_infos = [(step['observation'], step['info']) for step in episodes[i]]
                        observations, infos = zip(*obs_infos)
                        
                        frames = [observations_to_image(obs, info) for obs, info in obs_infos]
                        disp_info = {k: [info[k] for info in infos] for k in infos[0].keys()}

                        os.makedirs(os.path.join(self.config.bc.exp_folder, 'videos'), exist_ok=True)
                        
                        generate_video(
                            video_option=['disk'],
                            video_dir=vid_folder,
                            images=frames,
                            episode_id={stats['total_episodes']},
                            checkpoint_idx=300,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=None,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name
                        )
                    episodes[i] = []

            observations = next_observations
               


def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    
    # Add the 'cfg_path' argument
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the value of 'cfg_path' argument
    cfg_path = args.cfg_path
    
    trainer = BCTrainer(cfg_path)

    if trainer.config.bc.mode == 'train':
        trainer.train()
    else:
        trainer.eval_checkpoint(trainer.config.bc.eval.ckpt)
    # trainer.train()

if __name__ == "__main__":
    main()
    
    
    
