from collections.abc import Mapping
import numpy as np
from itertools import product
from tqdm import tqdm
from habitat.config.default_structured_configs import CollisionsMeasurementConfig, TopDownMapMeasurementConfig

from torch.distributed.elastic.multiprocessing.errors import record
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
from habitat_baselines.utils.common import batch_obs

from lmnav.common.utils import all_gather, all_reduce, catchtime, convert_weights_to_fp16, sum_dict
from lmnav.dataset.data_gen import _init_envs
from lmnav.common.episode_processor import apply_transforms_images


from habitat import logger
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
from lmnav.config.default import get_config
from lmnav.common.rollout_storage import RolloutStorage
from lmnav.models.base_policy import instantiate_model
from lmnav.common.lr_utils import get_lr_schedule_lambda

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (
    init_distrib_slurm,
    get_distrib_size,
    rank0_only,
)

import warnings

warnings.filterwarnings("ignore")


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
    def __init__(self, config, eval=False, verbose=False):
        self.config = config
        self.exp_folder = os.path.join(
            self.config.exp.root_dir,
            self.config.exp.group,
            self.config.exp.job_type,
            self.config.exp.name,
        )
        self.writer = instantiate(self.config.exp.logger, eval_mode=eval)
        self.verbose = verbose

    def validate_config(self):
        """
        validates that config parameters are constrained properly
        """
        pass

    def initialize_train(self):
        """
        Initialize distributed controller for DDP, starts data generator process
        """
        self.validate_config()
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
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
                logger.info(
                    f"Initialized DDP-PPO with {torch.distributed.get_world_size()} workers"
                )

            # update gpu ids for this process
            with read_write(self.config):
                self.config.device = f"cuda:{local_rank}"
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )
                self.config.habitat.task.measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=0,
                            map_resolution=128,
                            draw_source=False,
                            draw_border=False,
                            draw_shortest_path=False,
                            draw_view_points=False,
                            draw_goal_positions=False,
                            draw_goal_aabbs=False,
                        ),
                        "collisions": CollisionsMeasurementConfig(),
                    }
                )


            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)

            self.artifact_store = torch.distributed.PrefixStore("artifacts", tcp_store)

        os.makedirs(self.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, "ckpts"), exist_ok=True)

        self.envs, _ = _init_envs(self.config)

        if rank0_only():
            self.writer.open(self.config)

        self.model = self.setup_actor_critic()
        self.sample_generator = self.environment_sample_generator()

        # TODO; update img size to something from config
        self.rollouts = RolloutStorage(
            self.envs.num_envs,
            self.config.train.num_rollout_steps,
            self.model.module.actor.tokens_per_img,
            self.model.module.actor.hidden_size,
            device=self.device,
        )
        self.step = 0
        self.cumstats = {
            "step": 0,
            "learner/total_episodes_done": 0,
            "learner/total_episodes_successful": 0.0,
            "learner/total_success_rate": 0.0,
            "metrics/total_frames": 0,
        }

        actor_optim_params = list(
            filter(lambda p: p.requires_grad, self.model.module.actor.parameters())
        )
        critic_optim_params = list(
            filter(lambda p: p.requires_grad, self.model.module.critic.parameters())
        )
        self.optim = torch.optim.Adam(
            params=[
                {
                    "params": actor_optim_params,
                    "lr": self.config.train.lr_schedule.actor.lr,
                },
                {
                    "params": critic_optim_params,
                    "lr": self.config.train.lr_schedule.critic.lr,
                },
            ]
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optim,
            lr_lambda=[
                get_lr_schedule_lambda(self.config.train.lr_schedule.actor),
                get_lr_schedule_lambda(self.config.train.lr_schedule.critic),
            ],
        )
        if self.config.exp.resume_id is not None:
            ckpt_path = os.path.join(self.exp_folder, "ckpts", "latest.pth")
            self.load_checkpoint(ckpt_path)

        torch.distributed.barrier()

    def setup_actor_critic(self):
        """
        Sets up the actor and critic modules, wraps them in DDP if distributed is set
        """

        OmegaConf.resolve(self.config)
        model = instantiate_model(
            self.config.train.policy, writer=self.writer, store=self.artifact_store
        )
        model = model.to(self.device)


        self.vis_processor = model.vis_processor

        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)

        if rank0_only():
            num_params = sum([param.numel() for param in model.parameters()])
            num_trainable_params = sum(
                [param.numel() for param in model.parameters() if param.requires_grad]
            )
            print(
                f"Done setting up agent! Total params: {num_params}. Trainable Params: {num_trainable_params}"
            )

        return model

    def update_stats(self, episode_stats):
        stats_keys = sorted(episode_stats.keys())
        episode_stats = torch.tensor(
            [episode_stats[key] for key in stats_keys],
            device=self.device,
            dtype=torch.float32,
        )
        all_reduce(self.is_distributed, self.device, episode_stats)
        episode_stats /= self.world_size
        # write stats if on rank 0
        episode_stats = {k: episode_stats[i].item() for i, k in enumerate(stats_keys)}
        self.cumstats["step"] = self.step
        self.cumstats["metrics/total_frames"] += int(
            episode_stats["metrics/frames"] * self.world_size
        )
        self.cumstats["learner/total_episodes_done"] += int(
            episode_stats["learner/num_episodes_done"] * self.world_size
        )
        self.cumstats["learner/total_episodes_successful"] += int(
            episode_stats["learner/num_episodes_successful"] * self.world_size
        )
        self.cumstats["learner/total_success_rate"] = (
            (
                self.cumstats["learner/total_episodes_successful"]
                / self.cumstats["learner/total_episodes_done"]
            )
            if self.cumstats["learner/total_episodes_done"] > 0
            else 0.0
        )
        return {**self.cumstats, **episode_stats}

    def load_checkpoint(self, ckpt_path):
        # load checkpoint
        ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt_state_dict["model"], strict=False)
        self.optim.load_state_dict(ckpt_state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt_state_dict["lr_scheduler"])

        # TODO; check if saved and loaded configs are the same

        # update cum stats
        self.cumstats = ckpt_state_dict["stats"]
        self.step = self.cumstats["step"]

    def save_checkpoint(self, filename, archive_artifact=True):
        # only save parameters that have been updated
        param_grad_dict = {k: v.requires_grad for k, v in self.model.named_parameters()}

        state_dict = self.model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dict.keys() and not param_grad_dict[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": OmegaConf.to_container(self.config),
            "stats": self.cumstats,
        }

        ckpt_filepath = os.path.join(self.exp_folder, "ckpts", filename)
        torch.save(save_obj, ckpt_filepath)

        if archive_artifact:
            artifact_name = self.config.train.store_artifact.name
            self.writer.save_artifact(
                artifact_name, "model", os.path.abspath(ckpt_filepath)
            )

    def embed_observations(self, observations, goal_idxs_to_embed=None):
        observations = batch_obs(observations, self.device)
        rgbs, goals = map(
            lambda t: einops.rearrange(t, "b h w c -> b 1 c h w"),
            (observations["rgb"], observations["imagegoal"]),
        )
        if goal_idxs_to_embed is not None:
            goals = goals[goal_idxs_to_embed]
            
        return self.model.module.actor.embed_visual(rgbs, goals)


    def environment_sample_generator(self):
        # we use an abstract action generator fn so that different models
        # can preserve their state in the way that makes sense for them
        dones = [True for _ in range(self.envs.num_envs)]

        # initialize rollouts
        observations = self.envs.reset()
        with torch.no_grad(), self.model.no_sync():
            rgb_embds, goal_embds = self.embed_observations(observations)
        self.rollouts.insert(next_rgbs=rgb_embds[:, 0], next_goals=goal_embds[:, 0])

        self.model.eval()

        
        while True:
            with torch.no_grad(), self.model.no_sync():
                g_cpu = torch.Generator(device=self.device)
                g_cpu.manual_seed(0)
                action_generator = self.model.module.actor.fast_action_generator(
                    rollout_storage=self.rollouts,
                    sampler=instantiate(self.config.train.sampler),
                    use_cache=True,
                    max_its=self.config.train.num_rollout_steps,
                )

                it = range(self.config.train.num_rollout_steps)

                for i in it:
                    next(action_generator)
                    actions = action_generator.send(dones)

                    outputs = self.envs.step(actions)

                    next_observations, rewards, dones, infos = [
                        list(x) for x in zip(*outputs)
                    ]

                    # only embed goals if new episode has started
                    goal_idxs_to_embed = torch.tensor([i for i in range(len(dones)) if dones[i]], dtype=torch.long)
                    rgb_embds, new_goal_embds = self.embed_observations(next_observations, goal_idxs_to_embed=goal_idxs_to_embed)
                    goal_embds[goal_idxs_to_embed] = new_goal_embds

                    dones, rewards, actions = map(
                        lambda l: torch.tensor(l), (dones, rewards, actions)
                    )

                    successes = torch.tensor(
                        [info["success"] for info in infos], dtype=torch.bool
                    )
                    dtgs = torch.tensor(
                        [info["distance_to_goal"] for info in infos], dtype=torch.float
                    )
                    self.rollouts.insert(
                        next_rgbs=rgb_embds[:, 0],
                        next_goals=goal_embds[:, 0],
                        dones=dones,
                        rewards=rewards,
                        actions=actions,
                        successes=successes,
                        dtgs=dtgs
                    )

            del action_generator
            yield
            self.rollouts.reset()

    def compute_advantages(self, values, rewards):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.train.gamma * nextvalues - values[:, t]
            lastgaelam = (
                delta + self.config.train.gamma * self.config.train.lam * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = advantages.detach()
        return values, advantages, returns

    def batched_forward_pass(self, rgbs_t, goals_t, actions_t, mask_t, past_kv_cache_t, attn_mask_t):
        E, T = rgbs_t.shape[:2]
        num_minibatches = math.ceil(E / self.config.train.minibatch_size)
        minibatch_size = self.config.train.minibatch_size

        actions_t = actions_t.to(self.device)

        logits, values, logprobs = [], [], []
        for g in range(0, E, minibatch_size):
            minibatch = tuple(
                map(
                    lambda t: t[g : g + minibatch_size],
                    (rgbs_t, goals_t, actions_t, mask_t, past_kv_cache_t, attn_mask_t),
                )
            )
            minibatch_act_logits, minibatch_values, minibatch_logprobs = self.model(*minibatch)
            map(lambda t: t.to("cpu"), minibatch)

            logits.append(minibatch_act_logits)
            values.append(minibatch_values)
            logprobs.append(minibatch_logprobs)

        logits = torch.cat(logits)
        values = torch.cat(values).squeeze()  # get rid of critic 1 dim
        logprobs = torch.cat(logprobs)

        return logits, values, logprobs

    def clip_by_value(self, x, tensor_min, tensor_max):
        """
        Tensor extenstion to torch.clamp
        https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
        """
        clipped = torch.max(torch.min(x, tensor_max), tensor_min)
        return clipped

    def compute_loss(
        self, old_logprobs, values, logprobs, logits, vpreds, mask, advantages, returns
    ):
        vpredclipped = self.clip_by_value(
            vpreds,
            values - self.config.train.cliprange_value,
            values + self.config.train.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.train.cliprange, 1.0 + self.config.train.cliprange
        )

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
            "learner/loss": dict(
                policy=pg_loss.detach().item(),
                value=vf_loss.detach().item(),
                total=loss.detach().item(),
            ),
            "learner/policy": dict(
                entropy=entropy.detach().item(),
                approxkl=approxkl.detach().item(),
                policykl=policykl.detach().item(),
                clipfrac=pg_clipfrac.detach().item(),
                advantages_mean=masked_mean(advantages, mask).detach().item(),
                ratio=torch.mean(ratio.detach()).item(),
            ),
            "learner/returns": dict(
                mean=return_mean.detach().item(), var=return_var.detach().item()
            ),
            "learner/val": dict(
                vpred=masked_mean(vpreds, mask).detach().item(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach().item(),
                clipfrac=vf_clipfrac.detach().item(),
                mean=value_mean.detach().item(),
                var=value_var.detach().item(),
            ),
        }
        return pg_loss, self.config.train.vf_coef * vf_loss, flatten_dict(stats)

    def train_ppo_step(self):
        time_ppo_start = time.time()

        stats = {
            "metrics/frames": 0,
            "metrics/rollouts_wait_time": 0.0,
            "metrics/fps": 0.0,
            "learner/num_episodes_done": 0,
            "learner/num_episodes_successful": 0,
            "learner/distance_to_goal": 0.0,
            "learner/actor_lr": 0.0,
            "learner/critic_lr": 0.0,
        }

        # first collect environment samples
        next(self.sample_generator)

        (
            rgbs,
            goals,
            actions,
            rewards,
            dones,
            successes,
            dtgs,
            past_kv_cache
        ) = self.rollouts.generate_samples()
        time_rollouts_end = time.time()

        T = self.config.train.num_rollout_steps
        minibatch_size = self.config.train.minibatch_size
        num_grad_accums = self.config.train.num_grad_accums

        collected_frames = sum([len(t) for t in rgbs])
        assert collected_frames == T * self.envs.num_envs

        stats["metrics/frames"] = collected_frames
        stats["metrics/rollouts_wait_time"] = time_rollouts_end - time_ppo_start

        stats["learner/num_episodes_done"] = sum([torch.sum(d) for d in dones])
        stats["learner/num_episodes_successful"] = sum(
            [torch.sum(s) for s in successes]
        )
        stats["learner/distance_to_goal"] = sum([d[-1].item() for d in dtgs]) / len(dtgs)

        # construct batch
        mask_t = torch.stack(
            [
                torch.cat([torch.ones(t.shape[0]), torch.zeros(T - t.shape[0])])
                for t in rgbs
            ]
        )
        mask_t = mask_t.bool().to(self.device)
        rgbs_t = torch.stack(
            [F.pad(t, (0,) * 5 + (T - t.shape[0],), "constant", 0) for t in rgbs]
        )
        goals_t = torch.stack([g[0:1] for g in goals])
        actions_t = torch.stack(
            [F.pad(t, (0, T - t.shape[0]), "constant", 0) for t in actions]
        )
        rewards_t = torch.stack(
            [F.pad(t, (0, T - t.shape[0]), "constant", 0) for t in rewards]
        )
        max_past_kv_len = max([t.shape[-2] for t in past_kv_cache])
        past_kv_cache_t = torch.stack(
            [F.pad(t, (0, 0, max_past_kv_len - t.shape[-2], 0), "constant", 0) for t in past_kv_cache]
        ).to(torch.bfloat16)

        attn_mask_t = torch.ones(past_kv_cache_t.shape[0], max_past_kv_len, device=self.device)
        for i, t in enumerate(past_kv_cache):
            attn_mask_t[i, :max_past_kv_len - t.shape[-2]] = 0

        # gather all episodes from other gpus
        local_size = torch.tensor(attn_mask_t.shape[0], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_sizes, local_size)
        max_size = max(all_sizes).item()

        # pad all tensors to max size
        def pad_to_max(t):
            size_diff = max_size - t.shape[0]
            padding = torch.zeros((size_diff, *t.shape[1:]), device=t.device, dtype=t.dtype)
            return torch.cat([t, padding], dim=0)

        rgbs_t, goals_t, actions_t, rewards_t, mask_t, past_kv_cache_t, attn_mask_t = map(
            pad_to_max, (rgbs_t,goals_t,actions_t,rewards_t,mask_t,past_kv_cache_t,attn_mask_t)
        )


        with torch.no_grad(), self.model.no_sync():
            old_logits, values, old_logprobs = self.batched_forward_pass(
                rgbs_t, goals_t, actions_t, mask_t, 
                past_kv_cache_t, attn_mask_t
            )

            # TODO; add reference model kl penalty
            rewards_t = rewards_t.to(self.device)
            values, advantages, returns = self.compute_advantages(values, rewards_t)

        ls = local_size.item()
        batch_idxs = torch.cat([torch.randperm(ls), torch.randint(0, ls, (max_size - ls,))])
        E, T = rgbs_t.shape[:2]

        cum_train_stats_l = []
        for _, mb in product(
            range(self.config.train.ppo_epochs),
            range(0, E, num_grad_accums * minibatch_size),
        ):
            minibatches_left = math.ceil((E - mb) / minibatch_size)
            num_grad_steps = min(minibatches_left, num_grad_accums)

            for g in range(num_grad_steps):
                start_idx = mb + g * minibatch_size
                mb_idxs = batch_idxs[start_idx : start_idx + minibatch_size]
                minibatch = {
                    "rgbs": rgbs_t[mb_idxs],
                    "goals": goals_t[mb_idxs],
                    "actions": actions_t[mb_idxs],
                    "masks": mask_t[mb_idxs],
                    "advantages": advantages[mb_idxs],
                    "returns": returns[mb_idxs],
                    "logprobs": old_logprobs[mb_idxs],
                    "values": values[mb_idxs],
                    "past_kv_cache": past_kv_cache_t[mb_idxs],
                    "attn_mask": attn_mask_t[mb_idxs]
                }
                # move all to device
                minibatch = {k: v.to(self.device) for k, v in minibatch.items()}

                def backwards_loss():
                    logits, vpreds, logprobs = self.batched_forward_pass(
                        minibatch["rgbs"],
                        minibatch["goals"],
                        minibatch["actions"],
                        minibatch["masks"],
                        minibatch["past_kv_cache"],
                        minibatch["attn_mask"]
                    )
                    loss_p, loss_v, train_stats = self.compute_loss(
                        minibatch["logprobs"],
                        minibatch["values"],
                        logprobs,
                        logits,
                        vpreds,
                        minibatch["masks"],
                        minibatch["advantages"],
                        minibatch["returns"],
                    )
                    loss = loss_p + loss_v
                    loss.backward()
                    return train_stats

                if g < num_grad_steps - 1:
                    with self.model.no_sync():
                        train_stats = backwards_loss()
                else:
                    train_stats = backwards_loss()

                map(lambda k: minibatch[k].to("cpu"), minibatch.keys())
                cum_train_stats_l.append(train_stats)

            if self.config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.max_grad_norm
                )
            self.optim.step()
            self.optim.zero_grad()

        map(lambda t: t.to("cpu"), (mask_t, rgbs_t, goals_t, actions_t, rewards_t, past_kv_cache_t, attn_mask_t))
        dict_cum_train_stats = reduce(sum_dict, cum_train_stats_l)
        dict_cum_train_stats = {
            k: v / len(cum_train_stats_l) for k, v in dict_cum_train_stats.items()
        }

        time_ppo_end = time.time()
        stats["metrics/fps"] = stats["metrics/frames"] / (time_ppo_end - time_ppo_start)
        stats["learner/actor_lr"] = self.lr_scheduler.get_last_lr()[0]
        stats["learner/critic_lr"] = self.lr_scheduler.get_last_lr()[1]
        return {**dict_cum_train_stats, **stats}

    def train(self):
        self.initialize_train()

        with tqdm(total=self.config.train.steps) as pbar:
            while self.step < self.config.train.steps:
                stats = self.train_ppo_step()

                self.lr_scheduler.step()
                stats = self.update_stats(stats)

                torch.distributed.barrier()

                if rank0_only():
                    self.writer.write(stats)
                    if self.step % self.config.train.ckpt_freq == 0:
                        ckpt_num = self.step // self.config.train.ckpt_freq
                        self.save_checkpoint(f"ckpt.{ckpt_num}.pth", archive_artifact=True)
                    else:
                        self.save_checkpoint("latest.pth", archive_artifact=False)

                self.step += 1
                pbar.update(1)


@record
def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument("cfg_path", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--eval", action="store_true", help="Flag to enable evaluation mode"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Flag to enable debug mode"
    )
    parser.add_argument("--resume_run_id", type=str, help="Writer run id to restart")
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    resume_id = args.resume_run_id

    with read_write(config):
        config.exp.resume_id = resume_id
        config.habitat_baselines.num_environments = config.train.num_envs

        if args.eval:
            config.habitat_baselines.num_environments = config.eval.num_envs

    trainer = PPOTrainer(config, eval=args.eval, verbose=args.debug)

    if not args.eval:
        trainer.train()
    else:
        trainer.eval()


if __name__ == "__main__":
    main()
