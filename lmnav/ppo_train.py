from collections.abc import Mapping
from operator import add
from collections import Counter
from functools import reduce
import math
import os
import einops
import argparse
import torch
import random
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
import time

import numpy as np
from tqdm import tqdm
from habitat_baselines.utils.common import batch_obs

from pprint import pprint
from lmnav.common.utils import all_reduce
from lmnav.dataset.data_gen import  _init_envs
from lmnav.common.episode_processor import apply_transforms_inputs 
from lmnav.common.episode_processor import apply_transforms_actions, apply_transforms_images, apply_transforms_inputs, extract_inputs_from_dataset

from habitat import logger
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from lmnav.config.default import get_config
from lmnav.common.rollout_storage import RolloutStorage
from lmnav.common.registry import registry
from lmnav.models.ppo_agent import PPOAgent
from lmnav.models.linear_head import LinearHead
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

def flatten_dict(nested, sep="."):
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

    def __init__(self, config, verbose=False):
        self.config = config
        self.exp_folder = os.path.join(self.config.exp.root_dir,
                                       self.config.exp.group,
                                       self.config.exp.job_type,
                                       self.config.exp.name)
        self.writer = instantiate(self.config.exp.logger)
        self.verbose = verbose

    def validate_config(self):
        """
        validates that config parameters are constrained properly
        """ 
        batch_size = self.config.train.batch_size
        minibatch_size = self.config.train.minibatch_size
        num_minibatches = batch_size // minibatch_size
        num_grad_accums = self.config.train.num_grad_accums

        assert batch_size % minibatch_size == 0, 'batch must be evenly partitioned into minibatch sizes'
        assert num_minibatches % num_grad_accums == 0, '# of grad accums must divide num_minibatches equally'

    def initialize_train(self):
        """
        Initialize distributed controller for DDP, starts data generator process
        """
        self.validate_config()
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        local_rank, world_rank, world_size = get_distrib_size()
        self.world_size = world_size
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
         
            
        if rank0_only():
            self.writer.open(self.config)
            
        os.makedirs(self.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, 'ckpts'), exist_ok=True)
        
        self.envs, _ = _init_envs(self.config) 
        
        self.model = self.setup_actor_critic()
        self.sample_generator = self.environment_sample_generator()

        # TODO; update img size to something from config
        self.rollouts = RolloutStorage(self.envs.num_envs, self.config.train.num_rollout_steps)
        self.step = 0
        self.cumstats = {
            'step': 0,
            'metrics/total_frames': 0
        }


        actor_optim_params = list(filter(lambda p: p.requires_grad, self.model.module.actor.parameters()))
        critic_optim_params = list(filter(lambda p: p.requires_grad, self.model.module.critic.parameters()))
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
        actor_policy_cfg = self.config.train.actor
        # don't construct a policy from scratch, rather pull it from artifact
        if actor_policy_cfg.use_artifact_policy_config:
            assert actor_policy_cfg.load_artifact is not None, 'use_artifact_policy_config=True, but load_artifact not specified!'
            if rank0_only():
                ckpt_path = self.writer.load_model(actor_policy_cfg.load_artifact)
                self.artifact_store.set("actor_policy_ckpt", ckpt_path)
            else:
                self.artifact_store.wait(["actor_policy_ckpt"])
                ckpt_path = self.artifact_store.get("actor_policy_ckpt").decode('utf-8')
                
            print(f"Loading actor policy from config: {ckpt_path}")
            ckpt_state_dict = torch.load(ckpt_path, map_location=self.device)
            actor_policy_cfg = OmegaConf.create(ckpt_state_dict['config']).train.policy
        
        actor = instantiate(actor_policy_cfg)
        critic = instantiate(self.config.train.critic, in_dim=actor.hidden_size)
        
        model = PPOAgent(actor, critic)
        model = model.to(self.device)
        
        self.vis_processor = actor.vis_encoder.vis_processor
        
        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)

        if rank0_only():
            num_params = sum([param.numel() for param in model.parameters()])
            num_trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
            print(f"Done setting up agent! Total params: {num_params}. Trainable Params: {num_trainable_params}")
            
            params_with_gradients = [name for name, param in model.named_parameters() if param.requires_grad]
            # print("Params with gradients")
            # pprint(params_with_gradients)

        return model
        

    def update_stats(self, episode_stats):
        stats_keys = sorted(episode_stats.keys())
        episode_stats = torch.tensor([episode_stats[key] for key in stats_keys],
                                     device='cpu',
                                     dtype=torch.float32)
        episode_stats = all_reduce(self.is_distributed, self.device, episode_stats)
        episode_stats /= torch.distributed.get_world_size()

        episode_stats = { k: episode_stats[i].item() for i, k in enumerate(stats_keys) }

        self.cumstats['step'] = self.step
        self.cumstats['metrics/total_frames'] += int(episode_stats['metrics/frames'] * torch.distributed.get_world_size())

        return {
            **self.cumstats,
            **episode_stats
        }
    
 
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
            "stats": self.cumstats
        }

        ckpt_num = self.step // self.config.train.ckpt_freq
        ckpt_filepath = os.path.join(self.exp_folder, 'ckpts', f'ckpt.{ckpt_num}.pth')
        torch.save(save_obj, ckpt_filepath) 

        artifact_name = f'{self.config.exp.group}-{self.config.exp.job_type}-{self.config.exp.name}'
        artifact_name = artifact_name.replace('+', '_')
        artifact_name = artifact_name.replace('=', '_')
        self.writer.save_artifact(artifact_name, 'model', os.path.abspath(ckpt_filepath))

    
    def embed_observations(self, observations):
        observations = batch_obs(observations, self.device)
        rgbs, goals = map(lambda t: einops.rearrange(t, 'b h w c -> b 1 c h w'), (observations['rgb'], observations['imagegoal']))
        rgbs_t, goals_t = apply_transforms_images(self.vis_processor, rgbs, goals) 
        img_embds_t, img_atts_t = self.model.module.actor.embed_visual(torch.cat([rgbs_t, goals_t], dim=2).to(self.device))
        rgb_embds, goal_embds = img_embds_t[:, 0], img_embds_t[:, 1]

        map(lambda t: t.to('cpu'), (observations['rgb'], observations['imagegoal'], observations['depth']))
        del observations
        return rgb_embds, goal_embds

        
    def environment_sample_generator(self):
        # we use an abstract action generator fn so that different models
        # can preserve their state in the way that makes sense for them
        action_generator = self.model.module.actor.action_generator(self.envs.num_envs, self.config.train.deterministic)

        observations = self.envs.reset()
        with torch.no_grad():
            rgb_embds, goal_embds = self.embed_observations(observations)
        dones = [False for _ in range(self.envs.num_envs)]
        
        # insert initial obsevations
        self.rollouts.insert(next_rgbs=rgb_embds, next_goals=goal_embds)

        while True:
            with torch.no_grad():
                it = range(self.config.train.num_rollout_steps)
                it = tqdm(it, desc='generating rollout...') if self.verbose else it
                
                for _ in it:
                    next(action_generator) 
                    actions = action_generator.send((rgb_embds, goal_embds, dones))

                    outputs = self.envs.step(actions)
                    next_observations, rewards, dones, infos = [list(x) for x in zip(*outputs)] 
                    rgb_embds, goal_embds = self.embed_observations(next_observations) 
                    dones, rewards, actions = map(lambda l: torch.tensor(l), (dones, rewards, actions))
                    
                    self.rollouts.insert(next_rgbs=rgb_embds, next_goals=goal_embds, dones=dones, rewards=rewards, actions=actions) 
                    
            yield
            self.rollouts.reset()
            

    def compute_advantages(self, values, rewards):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.train.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.train.gamma * self.config.train.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = advantages.detach()
        return values, advantages, returns 

    
    def batched_forward_pass(self, rgbs_t, goals_t, actions_t, mask_t):
        E, T = rgbs_t.shape[:2]
        num_minibatches = math.ceil(E / self.config.train.minibatch_size)
        minibatch_size = self.config.train.minibatch_size

        actions_t = actions_t.to(self.device)
        
        logits, values, logprobs = [], [], []
        for g in range(0, E, minibatch_size):
            minibatch = tuple(map(lambda t: t[g: g + minibatch_size], (rgbs_t, goals_t, actions_t, mask_t)))
            minibatch_act_logits, minibatch_values, minibatch_logprobs = self.model(*minibatch)
            map(lambda t: t.to('cpu'), minibatch)
            
            logits.append(minibatch_act_logits)
            values.append(minibatch_values)
            logprobs.append(minibatch_logprobs)
        
        logits = torch.cat(logits) 
        values = torch.cat(values).squeeze() # get rid of critic 1 dim
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

        loss = pg_loss + self.config.train.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.train.ratio_threshold:
            print(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.train.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = {
            "learner/loss": dict(policy=pg_loss.detach().item(), value=vf_loss.detach().item(), total=loss.detach().item()),
            "learner/policy": dict(
                entropy=entropy.detach().item(),
                approxkl=approxkl.detach().item(),
                policykl=policykl.detach().item(),
                clipfrac=pg_clipfrac.detach().item(),
                advantages_mean=masked_mean(advantages, mask).detach().item(),
                ratio=torch.mean(ratio.detach()).item(),
            ),
            "learner/returns": dict(mean=return_mean.detach(), var=return_var.detach()),
            "learner/val": dict(
                vpred=masked_mean(vpreds, mask).detach().item(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach().item(),
                clipfrac=vf_clipfrac.detach().item(),
                mean=value_mean.detach().item(),
                var=value_var.detach().item(),
            ),
        }
        return pg_loss, self.config.train.vf_coef * vf_loss, flatten_dict(stats)
        
    def all_gather(self, q):
        """
        Gathers tensor arrays of different lengths across multiple gpus. 
        Assumes that q.shape[1:] is identical across gpus, and only pads dim=0
        
        Parameters
        ----------
            q : tensor array
            ws : world size
            device : current gpu device
            
        Returns
        -------
            all_q : list of gathered tensor arrays from all the gpus

        """
        q = q.to(self.device)
        local_size = torch.tensor(q.shape[0], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_sizes, local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros((size_diff, *q.shape[1:]), device=self.device, dtype=q.dtype)
            q = torch.cat((q, padding))

        all_qs_padded = [torch.zeros_like(q) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_qs_padded, q)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        all_qs = torch.cat(all_qs)
        return all_qs
    
    def train_ppo_step(self):
        
        stats = { 'metrics/frames': 0, 'metrics/rollouts_wait_time': 0. }
        
        # first collect environment samples
        rollouts_start_time = time.time()
        next(self.sample_generator)        
        rgbs, goals, actions, rewards = self.rollouts.generate_samples()
        rollouts_end_time = time.time()
        stats['metrics/rollouts_wait_time'] = rollouts_end_time - rollouts_start_time
        
        T = self.config.train.num_rollout_steps
        batch_size = self.config.train.batch_size
        minibatch_size = self.config.train.minibatch_size
        num_grad_accums = self.config.train.num_grad_accums
        
        collected_frames = sum([len(t) for t in rgbs])
        assert collected_frames == T * self.envs.num_envs

        # gather all episodes from other gpus
        stats['metrics/frames'] = collected_frames

        

        # construct batch
        mask_t = torch.stack([torch.cat([torch.ones(t.shape[0]), torch.zeros(T - t.shape[0])]) for t in rgbs])
        mask_t = mask_t.bool()
        rgbs_t = torch.stack([F.pad(t, (0,)*5 + (T - t.shape[0],), 'constant', 0) for t in rgbs])
        goals_t = torch.stack([g[0:1] for g in goals]) 
        actions_t = torch.stack([F.pad(t, (0, T - t.shape[0]), 'constant', 0) for t in actions])
        rewards_t = torch.stack([F.pad(t, (0, T - t.shape[0]), 'constant', 0) for t in rewards])

        
        start = time.time()
        rgbs_d, goals_d, actions_d, rewards_d, mask_d = map(lambda t: self.all_gather(t), (rgbs_t, goals_t, actions_t, rewards_t, mask_t)) 
        end = time.time()
        if rank0_only():
            print("Time taken to gather:", end - start)
            print("Shape of gathered tensor: ", rgbs_d.shape, self.device)
            
        # slice up the gathered data into equal sized pieces
        E = rgbs_d.shape[0] // self.world_size
        print(self.device, "gets", E)
        rgbs_t, goals_t, actions_t, rewards_t, mask_t = map(lambda t: t[self.rank * E:self.rank * E + E], (rgbs_d, goals_d, actions_d, rewards_d, mask_d))
        print(rgbs_t.shape, self.device)
        
        
        with torch.no_grad(), self.model.no_sync():
            old_logits, values, old_logprobs = self.batched_forward_pass(rgbs_t, goals_t, actions_t, mask_t) 

            # TODO; add reference model kl penalty
            rewards_t = rewards_t.to(self.device)
            values, advantages, returns = self.compute_advantages(values, rewards_t)
            batch_idxs = torch.randperm(len(rgbs))

        E, T = rgbs_t.shape[:2]
         
        cum_train_stats = {}
        for i in range(self.config.train.ppo_epochs):
            for mb in range(0, E, num_grad_accums * minibatch_size):                 
                minibatches_left = E - mb * minibatch_size
                for g in range(min(minibatches_left, num_grad_accums)):
                    mb_idxs = batch_idxs[mb + g: mb + g + minibatch_size]
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
                    # move all to device
                    minibatch = {k: v.to(self.device) for k, v in minibatch.items()}
                    

                    # TODO; add gradient parameter clipping
                    if g < num_grad_accums - 1:
                        with self.model.no_sync():
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
                            loss.backward() 
                    else:
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
                        loss.backward()

                    cum_train_stats = reduce(add, map(Counter, (cum_train_stats, train_stats))) 
                    
                self.optim.step()
                self.optim.zero_grad()

            
        
        mask_t.to('cpu')
        rgbs_t.to('cpu')
        goals_t.to('cpu')
        actions_t.to('cpu')
        rewards_t.to('cpu')

        return {
            **cum_train_stats,
            **stats
        }

   
    def train(self):
        self.initialize_train()

        while self.step < self.config.train.steps:
            stats = self.train_ppo_step()

            self.lr_scheduler.step()
            torch.distributed.barrier()
            stats = self.update_stats(stats)

            if rank0_only():
                self.writer.write(stats)
                if self.step % self.config.train.ckpt_freq == 0:
                    self.save_checkpoint()

            self.step += 1

            
def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    parser.add_argument('--eval', action='store_true', help='Flag to enable evaluation mode')
    parser.add_argument('--debug', action='store_true', help='Flag to enable debug mode')
    parser.add_argument('--resume_run_id', type=str, help="Writer run id to restart")
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    resume_id = args.resume_run_id

    with read_write(config):
        config.exp.resume_id = resume_id
        config.habitat_baselines.num_environments = config.train.num_envs

        if args.eval:
            config.habitat_baselines.num_environments = config.eval.num_envs
            config.exp.name = f'eval {config.exp.name}'
            config.exp.job_type = 'eval'

        if args.debug:
            config.exp.job_type = 'debug'
        
    trainer = PPOTrainer(config, verbose=args.debug)

    if not args.eval:
        trainer.train()
    else:
        trainer.eval()

if __name__ == "__main__":
    main()
    
    
    
