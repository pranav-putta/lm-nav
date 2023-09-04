import numpy as np
import random

from collections import namedtuple

import habitat
from habitat import logger
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (init_distrib_slurm, get_distrib_size, rank0_only)

import torch
import torch.distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import multiprocessing as mp

from typing import Any

from lmnav.data_gen import start_data_gen_process
from lmnav.common.config import Config as NavLLAMAConfig
from lmnav.models import *
from lmnav.processors import *
from lmnav.common.registry import registry as llama_registry
from lmnav.common.episode_processor import apply_transforms_inputs, extract_inputs_from_dataset, sample_subsequences


class BCTrainer:
    data_process: Any 
    data_conn: Any
    
    
    def initialize_train(self):
        """
        Initializes distributed controller for DDP, starts data generator process
        """
        self.config = habitat.get_config("lmnav/configs/habitat/imagenav_hm3d.yaml")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

            self.device = f"cuda:{local_rank}"
            
        # start data generator process
        self.data_process, self.data_conn = start_data_gen_process(self.device, self.config, deterministic=False)

        # set up student
        self.agent = self.setup_student()

        # set up optimizer
        optim_params = list(filter(lambda p: p.requires_grad, self.agent.parameters()))
        self.optim = torch.optim.Adam(params=optim_params, lr=self.config.bc.lr)
        self.dataset = []


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
        min_episodes = 10
        max_episodes = 20
        num_grad_accums = 4

        # accumulate data from the data queue; ideally the queue should already be filled
        self.dataset += [(self.data_conn.recv()) for _ in range(min_episodes)]
        
        if len(self.dataset) > max_episodes:
            self.dataset = self.dataset[len(self.dataset) - max_episodes:]

        print(f"Running behavior cloning over dataset of size: {len(self.dataset)}")
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

        print(f"Process {self.rank} epoch {epoch}. Average Loss: {avg_loss}")

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

            stats = { k: stats[i] for i, k in enumerate(stats_keys) }

            if rank0_only():
                # set stats to writer
                pass
            
            

def main():
    trainer = BCTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
    
    
    
