from collections.abc import Mapping
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

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy

def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat

def masked_mean(values, mask, axis=None):
        """Compute mean of tensor with a masked values."""
        if axis is not None:
            return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
        else:
            return (values * mask).sum() / mask.sum()
     

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
            
        
    def logprobs_from_logits(self, logits, labels):
        """
        See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
        """
        logp = F.log_softmax(logits, dim=2)
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy


    def compute_advantages(self, values, rewards):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = advantages.detach()
        return values, advantages, returns 

           
    
    def batched_forward_pass(self, rgbs_t, goals_t, actions_t, mask_t):
        num_minibatches = self.config.train.batch_size // self.config.train.minibatch_size
        minibatch_size = self.config.train.minibatch_size

        act_tkn_ids = self.model.actor.llama_tokenizer('stop forward left right', add_special_tokens=False, return_tensors='pt') 
        act_tkn_ids = act_tkn_ids.input_ids.to(self.device).squeeze()
        
        logits, values, logprobs = [], [], []
        for g in range(num_minibatches):
            minibatch = map(lambda t: t[g: g + minibatch_size], (rgbs_t, goals_t, actions_t, mask_t))
            minibatch_logits, minibatch_values = self.model(*minibatch) 
            act_positions = torch.tensor([-(self.model.actor.tokens_per_image + 1) * (T - i) + 2 for i in range(actions)]).to(self.device)
            minibatch_act_logits = minibatch_logits[:, act_positions, act_tkn_ids]

            minibatch_logprobs = self.logprobs_from_logits(minibatch_act_logits, actions_t)

            logits.append(minibatch_act_logits)
            values.append(minibatch_values)
            logprobs.append(minibatch_logprobs)
        
        logits = torch.cat(logits) 
        values = torch.cat(values)
        logprobs = torch.cat(logprobs)

        return logits, values, logprobs
        
    def clip_by_value(self, x, tensor_min, tensor_max):
        """
        Tensor extenstion to torch.clamp
        https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
        """
        clipped = torch.max(torch.min(x, tensor_max), tensor_min)
        return clipped
    
    def compute_loss(self,
                     old_logprobs,
                     values,
                     logprobs,
                     logits,
                     vpreds,
                     mask,
                     advantages,
                     returns):
        vpredclipped = self.clip_by_value(vpreds, values - self.config.train.cliprange_value, values + self.config.train.cliprange_value)

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.train.cliprange, 1.0 + self.config.train.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            print(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
        
    def train_epoch(self, epoch):
        # first collect environment samples
        next(self.action_generator)        
         
        rgbs, goals, actions, rewards = self.rollouts.generate_samples()
        T = self.config.train.max_rollout_steps
        batch_size = self.config.train.batch_size
        minibatch_size = self.config.train.minibatch_size
        assert batch_size % minibatch_size == 0, 'batch must be evenly partitioned into minibatche sizes'
        num_minibatches = batch_size // minibatch_size
        num_grad_accums = self.config.train.num_grad_accums
        assert minibatch_size % num_grad_accums == 0, '# of grad accums must divide minibatch equally'
        
        assert all([len(t) for t in rgbs]) == T

        rgbs_t, goals_t, actions_t, rewards_t = map(lambda t: torch.stack(t), (rgbs, goals, actions, rewards))
        rgbs_t, goals_t, actions_t = apply_transforms_inputs(self.vis_processor, rgbs_t, goals_t, actions_t)
        mask_t = torch.ones(*rgbs_t.shape[:2])

        old_logits, values, old_logprobs = self.batched_forward_pass(rgbs_t, goals_t, actions_t, mask_t) 
        stats = { 'loss': 0.0, 'num_steps': 0 }

        # TODO; add reference model kl penalty
        values, advantages, returns = self.compute_advantages(values, rewards_t)
        batch_idxs = torch.randperm(len(rgbs)).view(num_minibatches, minibatch_size)

        for _ in range(self.config.train.ppo_epochs):
            for mb in range(0, num_minibatches, num_grad_accums):                 
                for g in range(num_grad_accums):
                    mb_idxs = batch_idxs[mb + g]
                    minibatch = {
                        'rgbs': rgbs_t[mb_idxs],
                        'goals': goals_t[mb_idxs],
                        'actions': actions_t[mb_idxs],
                        'masks': mask_t[mb_idxs],
                        'advantages': advantages[mb_idxs],
                        'returns': returns[mb_idxs],
                        'logprobs': old_logprobs[mb_idxs],
                        'values': values[mb_idxs]
                    }
                    
                    logits, vpreds, logprobs = self.batched_forward_pass(minibatch['rgbs'], minibatch['goals'], minibatch['actions'], minibatch['masks'])
                    loss_p, loss_v, train_stats = self.compute_loss(
                        minibatch['logprobs'],
                        minibatch['values'],
                        logprobs,
                        logits,
                        vpreds,
                        minibatch['masks'],
                        minibatch['advantages'],
                        minibatch['returns']
                    ) 
                    loss = loss_p + loss_v
                    stats['loss'] += loss.cpu().item()

                    # TODO; add gradient parameterj clipping
                    if g < num_grad_accums - 1:
                        with self.model.no_sync():
                            loss.backward() 
                            pass
                    else:
                        loss.backward()
                        pass
                    
                self.optim.step()
                self.optim.zero_grad()
        
   
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
    
    
    
