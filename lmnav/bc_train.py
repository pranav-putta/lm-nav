import time
from habitat_baselines.utils.common import batch_obs
from habitat_sim.utils.datasets_download import argparse
import numpy as np
import random

from collections import namedtuple

import habitat
from habitat import logger
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (init_distrib_slurm, get_distrib_size, rank0_only)

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

        self.is_distributed = get_distrib_size()[2] > 1 or True

        if self.is_distributed:
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

            self.device = torch.device(f"cuda:{local_rank}")
            
        # start data generator process
        print(f"Now starting gen process on {self.rank}")
        self.data_process, self.data_conn = start_data_gen_process(self.device, self.config, deterministic=False)

        # set up student
        self.agent = self.setup_student()
        os.makedirs(self.config.bc.ckpt_folder, exist_ok=True)

        # set up optimizer
        optim_params = list(filter(lambda p: p.requires_grad, self.agent.parameters()))
        self.optim = torch.optim.Adam(params=optim_params, lr=self.config.bc.lr)
        self.dataset = []
        
        if rank0_only(): 
            self.writer = get_writer(self.config) 


    def setup_student(self):
        cfg_path = "/srv/flash1/pputta7/projects/lm-nav/exp_configs/lin_nav_llama_train.yaml"
        
        Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
        args = Args(cfg_path, "llama_v2", 0, [])

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
        print("Params with gradients")
        from pprint import pprint
        pprint(params_with_gradients)

        return agent

    
    def train_epoch(self, epoch):
        num_samples = 2
        max_state_length = 20
        num_bc_epochs = 4
        min_episodes = 5
        max_episodes = 20
        num_grad_accums = 4

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
        

    def train(self):
        self.initialize_train()

        epochs, batch_size = self.config.bc.epochs, self.config.bc.batch_size

        for epoch in range(epochs):
            stats = self.train_epoch(epoch)
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
                    ckpt_num = epoch // self.config.bc.ckpt_freq
                    torch.save(self.agent.state_dict(), os.path.join(self.config.bc.ckpt_folder, f'ckpt.{ckpt_num}.pth')) 
                    

                    
    def pad_sequences(self, seqs, dim):
        p2d_partial = (0,) * ((len(seqs[0].shape) - dim - 1) * 2 + 1)
        max_t = max([seq.shape[dim] for seq in seqs])
        
        padded_seqs = [F.pad(seq, p2d_partial + (max_t - seq.shape[dim],)) for seq in seqs]
        return torch.stack(padded_seqs)

    def eval_checkpoint(self, ckpt_path):
        N_episodes = 100
        T = 15
        # load checkpoint
        
        # start evaluation
        print(f"Starting evaluation of checkpoint {ckpt_path}")

        num_episodes_done = 0
        envs, env_spec = _init_envs(self.config)
        self.initialize_eval()

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

            import pdb

            # project onto the action space
            actions = []
            for i in range(len(partial_episodes)):
                pdb.set_trace()
                act_logits = logits[i, -act_pos_delta[i], tokens]
                act_logits = F.softmax(act_logits)
                actions.append(act_logits.argmax().cpu().item())
                
            pdb.set_trace()
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
        trainer.eval_checkpoint(None)
    # trainer.train()

if __name__ == "__main__":
    main()
    
    
    
