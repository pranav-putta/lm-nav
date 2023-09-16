import os
import argparse
import torch
import random
import torch.distributed
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

import numpy as np

from pprint import pprint
from lmnav.dataset.data_gen import  _init_envs
from lmnav.common.episode_processor import apply_transforms_inputs 

from habitat import logger
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from lmnav.config.default import get_config
from lmnav.common.rollout_storage import RolloutStorage
from lmnav.common.registry import registry
from lmnav.models.value_head import ValueHead
from lmnav.models.ppo_agent import PPOAgent
from lmnav.common.lr_utils import get_lr_schedule_lambda

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
        
        self.model = self.setup_actor_critic()
        self.action_generator = self.environment_sample_generator()

        # TODO; update img size to something from config
        self.rollouts = RolloutStorage(self.envs.num_envs, self.config.train.num_rollout_steps)
        self.epoch = 0
        self.writer = registry.get_logger_class(self.config.exp.logger._target_)(self.config)

        actor_optim_params = list(filter(lambda p: p.requires_grad, self.model.actor.parameters()))
        critic_optim_params = list(filter(lambda p: p.requires_grad, self.model.critic.parameters()))
        self.optim = torch.optim.Adam(params=[
            {'params': actor_optim_params, 'lr': self.config.train.lr_schedule.actor.lr},
            {'params': critic_optim_params, 'lr': self.config.train.lr_schedule.critic.lr}
        ])
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim,
                                                              lr_lambda=[get_lr_schedule_lambda(self.config.train.lr_schedule.actor),
                                                                         get_lr_schedule_lambda(self.config.train.lr_schedule.critic)])
        
        
    def setup_actor_critic(self):
        """
        Sets up the actor and critic modules, wraps them in DDP if distributed is set
        """
        actor = instantiate(self.config.train.policy.actor)
        critic = instantiate(self.config.train.policy.critic, in_dim=actor.hidden_size)
        model = PPOAgent(actor, critic)
        model = model.to(self.device)
        
        self.vis_processor = actor.vis_encoder.vis_processor
        
        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            model = DDP(model, device_ids=[self.rank])

        if rank0_only():
            num_params = sum([param.numel() for param in model.parameters()])
            num_trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
            print(f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}")
            
            params_with_gradients = [name for name, param in model.named_parameters() if param.requires_grad]
            print("Params with gradients")
            pprint(params_with_gradients)

        return model
        

    def _all_reduce(self, t):
        if not self.is_distributed:
            return t

        orig_device = t.device
        t = t.to(self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)
    
 
    def save_checkpoint(self):
        # only save parameters that have been updated
        param_grad_dict = {
            k: v.requires_grad for k, v in self.model.named_parameters()
        }

        state_dict = self.model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dict.keys() and not param_grad_dict[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": OmegaConf.to_container(self.config),
            "epoch": self.epoch
        }
        ckpt_num = self.epoch // self.config.train.ckpt_freq
        ckpt_filepath = os.path.join(self.exp_folder, 'ckpts', f'ckpt.{ckpt_num}.pth')
        torch.save(save_obj, ckpt_filepath) 

        self.writer.save_artifact(self.config.exp.name, 'model', os.path.abspath(ckpt_filepath))


    
    def environment_sample_generator(self):
        # we use an abstract action generator fn so that different models
        # can preserve their state in the way that makes sense for them
        action_generator = self.model.action_generator(self.envs.num_envs, self.config.train.deterministic)

        observations = self.envs.reset()
        dones = [False for _ in range(self.envs.num_envs)]
        
        # insert initial obsevations
        self.rollouts.insert(observations)
        
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
        rgbs, goals, actions, rewards = self.rollouts.generate_samples()
        p_idxs = torch.randperm(len(rgbs)).view(self.config.train.ppo_epochs,
                                                self.config.train.grad_accums,
                                                self.config.train.batch_size)

        total_loss = 0
        for ppo_epoch in range(self.config.train.ppo_epochs):
            for g in range(self.config.train.grad_accums):
                idxs = p_idxs[ppo_epoch, g]
                
                 # construct batch
                rgbs_t, goals_t, actions_t = map(lambda t: [t[i] for i in idxs], (rgbs, goals, actions))
                T = self.config.train.policy.actor.max_trajectory_length 
                # pad inputs to T
                mask_t = torch.stack([torch.cat([torch.ones(t.shape[0]), torch.zeros(T - t.shape[0])]) for t in rgbs_t])
                mask_t = mask_t.bool()
                rgbs_t = torch.stack([F.pad(t, (0,)*7 + (T - t.shape[0],), 'constant', 0) for t in rgbs_t])
                goals_t = torch.stack(goals_t) 
                actions_t = torch.stack([F.pad(t, (0, T - t.shape[0]), 'constant', 0) for t in actions_t])
                rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)

                if g < self.config.train.grad_accums - 1:
                    with self.model.no_sync():
                        # perform ppo update
                        outputs = self.model(rgbs_t, goals_t, actions_t, mask_t)
                        loss = outputs.loss
                        total_loss += loss.item()
                        loss.backward()
                        
                else:
                    outputs = self.model(rgbs_t, goals_t, actions_t, mask_t)
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
           
                rgbs_t.to('cpu')
                goals_t.to('cpu')
                    
            
            avg_loss = total_loss / (self.config.train.ppo_epochs * self.config.train.grad_accums)
            actor_lr = self.lr_scheduler.get_last_lr()[0]
            critic_lr = self.lr_scheduler.get_last_lr()[1]
            
            return {
                'loss': avg_loss,
                'epoch': epoch,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr
            }
    
    
    def train(self):
        self.initialize_train()
        
        epochs, batch_size = self.config.train.epochs, self.config.train.batch_size

        while self.epoch < epochs:
            stats = self.train_epoch(self.epoch)
            self.lr_scheduler.step()
            
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

                if self.epoch % self.config.train.ckpt_freq == 0:
                    self.save_checkpoint()

            self.epoch += 1

            
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
    
    
    
