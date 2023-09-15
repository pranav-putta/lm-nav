import argparse
import os
import torch
import random
import torch.distributed
import numpy as np

from pprint import pprint
from lmnav.dataset.data_gen import  _init_envs

from habitat import logger
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from lmnav.config.default import get_config
from lmnav.common.rollout_storage import RolloutStorage

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (init_distrib_slurm, get_distrib_size, rank0_only)

class PPOTrainer:

    def __init__(self, config):
        self.config = config
        self.exp_folder = os.path.join(self.config.exp.root_dir, self.config.exp.name)

    def initialize_train(self):
        """
        Initialize distributed controller for DDP, starts data generator process
        """
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
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
                self.config.device = f'cuda:{local_rank}'
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
        
        self.envs, _ = _init_envs(self.config) 
        
        self.actor = self.setup_actor()
        self.critic = self.setup_critic()
        self.action_generator = self.environment_sample_generator()

        # TODO; update img size to something from config
        self.rollouts = RolloutStorage(self.envs.num_envs, self.config.train.num_rollout_steps, 224)
       

    def setup_actor(self):
        model = instantiate(self.config.train.policy)

        self.vis_processor = model.vis_encoder.vis_processor

        actor = model.to(self.device)
        actor.train()
        
        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            actor = DDP(actor, device_ids=[self.rank])

        num_params = sum([param.numel() for param in actor.parameters()])
        num_trainable_params = sum([param.numel() for param in actor.parameters() if param.requires_grad])
        
        print(f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}")
        
        params_with_gradients = [name for name, param in model.named_parameters() if param.requires_grad]
        if rank0_only():
            print("Params with gradients")
            pprint(params_with_gradients)

        return actor

    
    def setup_critic(self):
        return None
    
    def environment_sample_generator(self):
        # we use an abstract action generator fn so that different models
        # can preserve their state in the way that makes sense for them
        action_generator = self.actor.action_generator(envs.num_envs, self.config.train.deterministic)

        observations = self.envs.reset()
        dones = [False for _ in range(self.envs.num_envs)]
        
        while True:
            with torch.no_grad():
                for _ in range(self.config.train.max_steps):
                    next(action_generator) 
                    actions = action_generator.send((observations, dones))

                    outputs = self.envs.step(actions)
                    next_observations, rewards, dones, infos = [list(x) for x in zip(*outputs)] 

                    self.rollouts.insert(next_observations, dones=dones, rewards=rewards, actions=actions) 
                    
                    observations = next_observations
                yield
            
        
    def train_epoch(self, epoch):
        
        # first collect environment samples
        next(self.action_generator)        
         
        # then compute ppo loss and backprop
        for ppo_epoch in range(self.config.train.ppo_epochs):
            
            

    

def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    parser.add_argument('--eval', action='store_true', help='Flag to enable evaluation mode')
    parser.add_argument('--resume_run_id', type=str, help="Writer run id to restart")
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    resume_id = args.resume_run_id

    with read_write(config):
        config.exp.logger.resume_id = resume_id
        config.habitat_baselines.num_environments = config.train.num_envs
        
    trainer = PPOTrainer(config)

    if not args.eval:
        trainer.train()
    else:
        with read_write(config):
            config.habitat_baselines.wb.group = 'eval'
            config.habitat_baselines.wb.run_name = f'eval {config.habitat_baselines.wb.run_name}'
            config.habitat_baselines.num_environments = config.eval.num_envs
            config.exp.logger.group = 'eval'
            # config.habitat.dataset.split = 'val_hard'
     
        trainer.eval()


if __name__ == "__main__":
    main()
    
    
    
