import numpy as np
import random

from collections import namedtuple

import habitat
from habitat import logger
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (init_distrib_slurm, get_distrib_size, rank0_only)

import torch
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
            local_rank, tcp_store = init_distrib_slurm(self.config.habitat_baselines.rl.ddppo.distrib_backend)
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
        import pdb
        pdb.set_trace()
        agent = DDP(agent, device_ids=[self.rank])

        num_params = sum([param.numel() for param in agent.parameters()])
        num_trainable_params = sum([param.numel() for param in agent.parameters() if param.requires_grad])
        
        print(f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}")
        return agent

    
    def train_epoch(self, epoch):
        num_samples = 100 
        max_state_length = 20
        num_bc_epochs = 4
        min_episodes = 2
        max_episodes = 1000

        # accumulate data from the data queue; ideally the queue should already be filled
        while self.data_conn.poll() or len(self.dataset) < min_episodes:
            self.dataset.append(self.data_conn.recv())
        
        if len(self.dataset) > max_episodes:
            self.dataset = self.dataset[len(self.dataset) - max_episodes:]

        print(f"Running behavior cloning over dataset of size: {len(self.dataset)}")
        rgbs, goals, actions = extract_inputs_from_dataset(self.dataset)

        for bc_epoch in range(num_bc_epochs):
            rgbs_t, goals_t, actions_t = sample_subsequences(num_samples, max_state_length, rgbs, goals, actions)
            rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)
            print("Shapes:", rgbs_t.shape, goals_t.shape, actions_t.shape)

            
            self.optim.zero_grad()
            
            outputs = self.agent(rgbs_t, goals_t, actions_t)
            print(outputs.loss)

            loss = outputs.loss
            loss.backward()

            self.optim.step()
            
        print(f"Process {self.rank} finished with epoch {epoch}")
        print()
        
        
    def train(self):
        self.initialize_train()

        epochs, batch_size = self.config.bc.epochs, self.config.bc.batch_size

        for epoch in range(epochs):
            self.train_epoch(epoch)
            

def main():
    trainer = BCTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
    
    
    
