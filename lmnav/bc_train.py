import math

import pdb
import einops
import time

from pprint import pprint
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_sim.utils.datasets_download import argparse
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from lmnav.common.utils import catchtime, forward_minibatches, levenshtein_distance

import numpy as np
import random
from hydra.utils import instantiate

from habitat import logger
from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import (
    init_distrib_slurm,
    get_distrib_size,
    rank0_only,
)
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from lmnav.common.lr_utils import get_lr_schedule_lambda
from lmnav.common.resumable_random_sampler import DistributedResumableSampler
from lmnav.config.default import get_config
from lmnav.common.utils import all_gather, all_reduce

from lmnav.models import *
from lmnav.dataset.datasets import OfflineEpisodeDataset
from lmnav.models.base_policy import instantiate_model
from lmnav.processors import *
from lmnav.common.episode_processor import (
    construct_subsequences,
)

from lmnav.common.writer import *


os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class BCTrainRunner:
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.exp_folder = os.path.join(
            self.config.exp.root_dir,
            self.config.exp.group,
            self.config.exp.job_type,
            self.config.exp.name,
        )
        self.writer = instantiate(self.config.exp.logger)

    def validate_config(self):
        """
        validates that config parameters are constrained properly
        """
        batch_size = self.config.train.batch_size
        minibatch_size = self.config.train.minibatch_size
        num_minibatches = batch_size // minibatch_size
        num_grad_accums = self.config.train.num_grad_accums

        assert (
            batch_size % minibatch_size == 0
        ), "batch must be evenly partitioned into minibatch sizes"
        assert (
            num_minibatches % num_grad_accums == 0
        ), "# of grad accums must divide num_minibatches equally"

    def initialize_train(self):
        """
        Initializes distributed controller for DDP, starts data generator process
        """
        self.validate_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.world_size = world_size

            if rank0_only():
                logger.info(
                    f"Initialized DDP-BC with {torch.distributed.get_world_size()} workers"
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

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)

            self.artifact_store = torch.distributed.PrefixStore("artifacts", tcp_store)

        # set up student
        os.makedirs(self.exp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.exp_folder, "ckpts"), exist_ok=True)

        # set up writer and scatter all relevant data to worker nodes
        if rank0_only():
            self.writer.open(self.config)
            data_files = self.writer.load_dataset(self.config.train.dataset)
            self.artifact_store.set("data_files", ";".join(data_files))
        else:
            self.artifact_store.wait(["data_files"])
            data_files = (
                self.artifact_store.get("data_files").decode("utf-8").split(";")
            )

        # set up dataset
        self.transforms = instantiate(self.config.train.transforms)
        self.dataset = OfflineEpisodeDataset(self.transforms, files=data_files)

        self.sampler = DistributedResumableSampler(
            self.dataset, self.rank, self.world_size
        )
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.train.episodes_per_batch,
            collate_fn=lambda x: x,
            num_workers=1,
            sampler=self.sampler,
        )

        self.agent = self.setup_student()

        # set up optimizer
        optim_groups = self.agent.module.configure_optim_groups()
        optim_groups = [
            {**group, "lr": self.config.train.lr_schedule.lr} for group in optim_groups
        ]
        self.optim = torch.optim.Adam(params=optim_groups)

        # set up lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optim,
            lr_lambda=[
                get_lr_schedule_lambda(self.config.train.lr_schedule)
                for _ in range(len(optim_groups))
            ],
        )

        self.step, self.epoch = 0, 0
        self.cumstats = {"step": 0, "epoch": 0, "metrics/total_frames": 0}

        if self.config.exp.resume_id is not None:
            ckpt_path = os.path.join(self.exp_folder, "ckpts", "latest.pth")
            self.load_checkpoint(ckpt_path)

        torch.distributed.barrier()
        print(f"Process {self.rank}: Ready")
        if rank0_only():
            print("Starting train!")

    def setup_student(self):
        OmegaConf.resolve(self.config)
        model = instantiate_model(
            self.config.train.policy, writer=self.writer, store=self.artifact_store
        )

        self.vis_encoder = model.vis_encoder

        agent = model.to(self.device)
        agent.train()

        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            agent = DDP(agent, device_ids=[self.rank], find_unused_parameters=True)

        num_params = sum([param.numel() for param in agent.parameters()])
        num_trainable_params = sum(
            [param.numel() for param in agent.parameters() if param.requires_grad]
        )

        print(
            f"Done setting up student! Total params: {num_params}. Trainable Params: {num_trainable_params}"
        )

        params_with_gradients = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        if rank0_only():
            print("Params with gradients")
            pprint(params_with_gradients)

        return agent

    def update_stats(self, episode_stats):
        stats_keys = sorted(episode_stats.keys())
        episode_stats = torch.tensor(
            [episode_stats[key] for key in stats_keys],
            device="cpu",
            dtype=torch.float32,
        )
        episode_stats = all_reduce(self.is_distributed, self.device, episode_stats)
        episode_stats /= torch.distributed.get_world_size()

        episode_stats = {k: episode_stats[i].item() for i, k in enumerate(stats_keys)}

        self.cumstats["step"] = self.step
        self.cumstats["epoch"] = self.epoch
        self.cumstats["metrics/total_frames"] += int(
            episode_stats["metrics/frames"] * torch.distributed.get_world_size()
        )

        return {**self.cumstats, **episode_stats}

    def train_bc_step(self, episodes):
        T = self.config.train.policy.max_trajectory_length
        batch_size = self.config.train.batch_size
        minibatch_size = self.config.train.minibatch_size
        num_minibatches = batch_size // minibatch_size
        num_grad_accums = self.config.train.num_grad_accums

        stats = {
            "learner/loss": 0.0,
            "metrics/frames": 0,
            "metrics/fps": 0.0,
            "learner/lr": 0,
            "learner/edit_distance": 0,
        }

        start_time = time.time()

        def forward_backwards_model(rgbs_t, goals_t, actions_t, mask_t):
            outputs = self.agent(rgbs_t, goals_t, actions_t, mask_t)
            loss, logits = outputs.loss, outputs.logits
            probs = F.softmax(logits, dim=-1)

            stats["learner/loss"] += loss.item()

            # compute levenshtein distances
            # distances = []
            # for i in range(mask_t.shape[0]):
            # a = torch.argmax(probs[i, : mask_t[i].sum()], dim=-1)
            # b = actions_t[i, : mask_t[i].sum()]
            # distances.append(levenshtein_distance(a, b))
            # stats["learner/edit_distance"] += sum(distances) / len(distances)

            loss.backward()

        actions = [episode["action"] for episode in episodes]
        goals = [episode["imagegoal"] for episode in episodes]
        rgbs = [episode["rgb"] for episode in episodes]

        # construct subsequences
        rgbs, goals, actions = construct_subsequences(
            batch_size, T, rgbs, goals, actions
        )

        # pad sequences
        mask = torch.stack(
            [
                torch.cat([torch.ones(t.shape[0]), torch.zeros(T - t.shape[0])])
                for t in rgbs
            ]
        ).bool()
        rgbs = torch.stack(
            [F.pad(t, (0, 0, 0, 0, 0, T - t.shape[0]), "constant", 0) for t in rgbs]
        )
        goals = torch.stack(goals)
        actions = torch.stack(
            [F.pad(t, (0, T - t.shape[0]), "constant", 0) for t in actions]
        )

        # to ensure equal batch sizes across gpus, gather them all
        rgbs, goals, actions, mask = map(
            lambda t: all_gather(t.contiguous(), self.device, self.world_size),
            (rgbs, goals, actions, mask),
        )
        E = rgbs.shape[0] // self.world_size

        # TODO; rename these variables to improve readability
        rgbs, goals, actions, mask = map(
            lambda t: t[self.rank * E : self.rank * E + E].clone(),
            (rgbs, goals, actions, mask),
        )
        batch_idxs = torch.randperm(E)

        for mb in range(0, E, num_grad_accums * minibatch_size):
            minibatches_left = math.ceil((E - mb) / minibatch_size)
            num_grad_steps = min(minibatches_left, num_grad_accums)
            for g in range(num_grad_steps):
                start_idx = mb + g * minibatch_size
                mb_idxs = batch_idxs[start_idx : start_idx + minibatch_size]
                # construct batch
                rgbs_t, goals_t, actions_t, mask_t = map(
                    lambda t: t[mb_idxs], (rgbs, goals, actions, mask)
                )
                if g < num_grad_accums - 1:
                    with self.agent.no_sync():
                        forward_backwards_model(rgbs_t, goals_t, actions_t, mask_t)
                else:
                    forward_backwards_model(rgbs_t, goals_t, actions_t, mask_t)
                rgbs_t.to("cpu")
                goals_t.to("cpu")

            if self.config.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.train.max_grad_norm
                )
            self.optim.step()
            self.optim.zero_grad()

        stats["learner/loss"] /= num_minibatches
        stats["learner/lr"] = self.lr_scheduler.get_last_lr()[0]
        stats["metrics/frames"] = sum([episode["rgb"].shape[0] for episode in episodes])

        rgbs.to("cpu")
        goals.to("cpu")

        torch.cuda.empty_cache()
        end_time = time.time()

        stats["frames/fps"] = stats["metrics/frames"] / (end_time - start_time)

        return stats

    def load_checkpoint(self, ckpt_path):
        # load checkpoint
        print(f"Loading model from checkpoint")
        ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")
        self.agent.load_state_dict(ckpt_state_dict["model"], strict=False)
        self.optim.load_state_dict(ckpt_state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt_state_dict["lr_scheduler"])
        self.sampler.load_state_dict(ckpt_state_dict["sampler"])

        # TODO; check if saved and loaded configs are the same

        # update cum stats
        self.cumstats = ckpt_state_dict["stats"]
        self.step = self.cumstats["step"]
        self.epoch = self.cumstats["epoch"]

    def save_checkpoint(self, filename, archive_artifact=True):
        # only save parameters that have been updated
        param_grad_dict = {k: v.requires_grad for k, v in self.agent.named_parameters()}

        state_dict = self.agent.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dict.keys() and not param_grad_dict[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optim.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "sampler": self.sampler.state_dict(),
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

    def train(self):
        self.initialize_train()

        while self.step < self.config.train.steps:
            for batch in self.data_loader:
                stats = self.train_bc_step(batch)

                self.lr_scheduler.step()
                torch.distributed.barrier()
                stats = self.update_stats(stats)

                if rank0_only():
                    self.writer.write(stats)
                    if self.step % self.config.train.ckpt_freq == 0:
                        ckpt_num = self.step // self.config.train.ckpt_freq
                        self.save_checkpoint(
                            f"ckpt.{ckpt_num}.pth", archive_artifact=True
                        )
                    else:
                        self.save_checkpoint("latest.pth", archive_artifact=False)

                self.step += 1

            self.epoch += 1


def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument("cfg_path", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--debug", action="store_true", help="Flag to enable debug mode"
    )
    parser.add_argument("--resume_run_id", type=str, help="Writer run id to restart")
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    resume_id = args.resume_run_id

    with read_write(config):
        config.exp.resume_id = resume_id

    trainer = BCTrainRunner(config, verbose=args.debug)
    trainer.train()


if __name__ == "__main__":
    main()
