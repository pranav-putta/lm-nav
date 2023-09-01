import numpy as np
import random

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

class BCTrainer:
    data_process: Any 
    data_queue: mp.Queue
    
    
    def initialize_train(self):
        """
        Initializes distributed controller for DDP, starts data generator process
        """
        self.config = habitat.get_config("config/imagenav_hm3d.yaml")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_distributed = get_distrib_size()[2] > 1

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
        self.data_process, self.data_queue = start_data_gen_process(self.device, self.config, deterministic=False)

        # set up student
        self.agent = self.setup_student()

        # set up optimizer
        optim_params = list(filter(lambda p: p.requires_grad, self.agent.parameters()))
        self.optim = torch.optim.Adam(params=optim_params, lr=self.config.bc.lr)


    def setup_student(self):
        # TODO: set this up actually lol
        agent = nn.Linear(100, 1)
        agent = agent.to(self.device)
        agent = DDP(agent, device_ids=[self.rank])
        return agent

    
    def train_epoch(self, epoch):
        batch_size = self.config.bc.batch_size 

        # accumulate data from the data queue; ideally the queue should already be filled
        data = [self.data_queue.get() for _ in range(batch_size)]

        loss_fn = nn.MSELoss()
        X = torch.ones((10, 100), device=self.device)
        Y = torch.ones((10, 1), device=self.device)

        self.optim.zero_grad()
        outputs = self.agent(X)
        loss = loss_fn(outputs, Y).backward()
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
    
    
    
