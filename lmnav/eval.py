import pickle

import einops

from pprint import pprint
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_sim.utils.datasets_download import argparse
from habitat_baselines.utils.info_dict import extract_scalars_from_info

from hydra.utils import instantiate

from habitat.config import read_write
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only

import torch
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from lmnav.common.rollout_storage import RolloutStorage
from lmnav.common.utils import create_mask

from lmnav.config.default import get_config

from lmnav.dataset.data_gen import _init_envs
from lmnav.models import *
from lmnav.models.base_policy import instantiate_model
from lmnav.processors import *
from lmnav.common.episode_processor import apply_transforms_images

from lmnav.common.writer import *

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class EvalRunner:
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.exp_folder = os.path.join(
            self.config.exp.root_dir,
            self.config.exp.group,
            self.config.exp.job_type,
            self.config.exp.name,
        )
        self.writer = instantiate(self.config.exp.logger, eval_mode=True)

    def current_episodes(self):
        for write_fn in self.envs._connection_write_fns:
            write_fn(("call", ("current_episode", {"all_info": True})))
        results = []
        for read_fn in self.envs._connection_read_fns:
            results.append(read_fn())
        return results

    def auto_find_resume_run(self):
        runs = self.writer.load_runs(
            filters={
                "config.exp.name": self.config.exp.name,
                "config.exp.group": self.config.exp.group,
                "config.exp.job_type": self.config.exp.job_type,
            }
        )
        # filter for eval runs
        runs = list(filter(lambda r: r.job_type == "eval", runs))
        assert (
            len(runs) <= 1
        ), "found multiple eval jobs for this run. not sure what to do"

        if len(runs) == 1:
            with read_write(self.config):
                self.config.exp.resume_id = runs[0].id

    def initialize_eval(self):
        """
        Initializes controller for evaluation process.
        NOTE: distributed eval is not set up here
        """
        self.validate_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = 0
        self.is_distributed = False
        self.eval_dir = os.path.join(self.exp_folder, "eval")

        self.auto_find_resume_run()
        self.writer.open(
            self.config,
            override_run_name=f"eval {os.path.join(self.config.exp.job_type,self.config.exp.name)}",
        )

        self.envs, env_spec = _init_envs(self.config)
        self.agent = self.setup_student()
        self.agent.eval()

        self.tablecols = [
            "ckpt",
            "episode_id",
            "difficulty",
            "geodesic_distance",
            "success",
        ]
        self.stats_path = os.path.join(self.eval_dir, "stats.pkl")

        self.table_data = []
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "rb") as f:
                self.table_data = pickle.load(f)

    def validate_config(self):
        pass

    def setup_student(self):
        OmegaConf.resolve(self.config)
        model = instantiate_model(
            self.config.eval.policy, writer=self.writer, store=None
        )

        self.vis_processor = model.vis_processor

        agent = model.to(self.device)
        agent.train()

        if self.is_distributed:
            print(f"Setting up DDP on GPU {self.rank}")
            agent = DDP(agent, device_ids=[self.rank])

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

    def load_checkpoint(self, ckpt_path):
        print(f"Loading model from checkpoint: {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path)
        ckpt_state_dict = {
            k[len("module.") :]: v for k, v in ckpt_state_dict["model"].items()
        }
        self.agent.load_state_dict(ckpt_state_dict, strict=False)
        self.agent.eval()

    def save_episode_video(self, episode, num_episodes, video_dir, ckpt_idx):
        _, infos = zip(*episode)

        frames = [observations_to_image(obs, info) for obs, info in episode]
        disp_info = {k: [info[k] for info in infos] for k in infos[0].keys()}

        try:
            generate_video(
                video_option=["disk"],
                video_dir=video_dir,
                images=frames,
                episode_id=num_episodes,
                checkpoint_idx=ckpt_idx,
                metrics=extract_scalars_from_info(disp_info),
                fps=self.config.habitat_baselines.video_fps,
                tb_writer=None,
                keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
            )
        except Exception as e:
            print(f"Error generating video: {e}")

    def eval(self):
        self.initialize_eval()

        if self.config.eval.policy.load_artifact.version == "*":
            versions = self.writer.load_model_versions(
                self.config.eval.policy.load_artifact
            )
            # eval versions from latest to earliest
            versions = reversed(sorted(versions, key=lambda t: int(t[1:])))
        else:
            versions = [self.config.eval.policy.load_artifact.version]

        for version in versions:
            with read_write(self.config):
                self.config.eval.policy.load_artifact.version = version
            ckpt_path = self.writer.load_model(self.config.eval.policy.load_artifact)
            self.eval_checkpoint(ckpt_path)

    def embed_observations(self, observations, goal_idxs_to_embed=None):
        observations = batch_obs(observations, self.device)
        rgbs, goals = map(
            lambda t: einops.rearrange(t, "b h w c -> b 1 c h w"),
            (observations["rgb"], observations["imagegoal"]),
        )
        if goal_idxs_to_embed is not None:
            goals = goals[goal_idxs_to_embed]
            
        return self.agent.embed_visual(rgbs, goals)

    @torch.inference_mode()
    def eval_checkpoint(self, ckpt_path):
        print(f"Starting evaluation for {ckpt_path}")
        # TODO; some parameters are constant, pull from config in the future
        buffer_length = self.config.habitat.environment.max_episode_steps
        self.rollouts = RolloutStorage(
                self.envs.num_envs,
                buffer_length,
                1,
                4096,
                device=self.device,
        )

        # some bookkeeping
        ckpt_name = os.path.basename(ckpt_path)
        eval_dir = os.path.join(self.eval_dir, ckpt_name)
        video_dir = os.path.join(eval_dir, "videos")
        os.makedirs(eval_dir, exist_ok=True)

        episode_data = [[] for _ in range(self.envs.num_envs)]

        if self.config.eval.save_videos:
            os.makedirs(video_dir, exist_ok=True)

        # load checkpoint
        self.load_checkpoint(ckpt_path)

        # turn of all gradients
        self.agent.eval()
        for param in self.agent.parameters():
            param.requires_grad = False


        # we use an abstract action generator fn so that different models
        # can preserve their state in the way that makes sense for them
        dones = [True for _ in range(self.envs.num_envs)]
        stats = {}

        # initialize rollouts
        observations = self.envs.reset()
        with torch.inference_mode():
            rgb_embds, goal_embds = self.embed_observations(observations)
        self.rollouts.insert(next_rgbs=rgb_embds[:, 0], next_goals=goal_embds[:, 0])
        sampler = instantiate(self.config.eval.sampler)

        num_episodes_done = 0
        
        pbar = tqdm(total=self.config.eval.num_episodes)
        action_generator = self.agent.action_generator(
            rollouts=self.rollouts,
            sampler=sampler,
        )

        while num_episodes_done < self.config.eval.num_episodes:
            for _ in tqdm(range(buffer_length), leave=False, desc="doing rollout..."):
                next(action_generator)
                actions, logprobs, hx = action_generator.send(dones)

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
                    dtgs=dtgs,
                    hx=hx,
                    logprobs=logprobs
                )

                for env in range(self.envs.num_envs):
                    episode_data[env].append((observations[env], infos[env]))
                    if dones[env]:
                        num_episodes_done += 1
                        self.save_episode_video(episode_data[env], num_episodes_done, video_dir, ckpt_name)
                        episode_data[env] = []

                observations = next_observations

            # compute stats
            batch, _ = self.rollouts.generate_samples()
            lengths = torch.tensor([episode['rgb'].shape[0] for episode in batch], device=self.device)
            batch = self.rollouts.pad_samples(batch)
            batch['mask'] = create_mask(lengths)
            stats["learner/num_episodes_done"] = (batch['done'] * batch['mask']).sum().item()
            stats["learner/num_episodes_successful"] = (batch['success'] * batch['mask']).sum().item()
            
            pbar.update(stats["learner/num_episodes_done"])
            self.rollouts.reset()

def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument("cfg_path", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--debug", action="store_true", help="Flag to enable debug mode"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Flag to quickly enable determinism",
    )
    parser.add_argument("--version", type=str, help="Which version of the model to run")
    args = parser.parse_args()

    config = get_config(args.cfg_path)

    with read_write(config):
        config.habitat_baselines.num_environments = config.eval.num_envs
        config.eval.deterministic = args.deterministic
        if args.version:
            config.eval.policy.load_artifact.version = args.version

    runner = EvalRunner(config, verbose=args.debug)
    runner.eval()


if __name__ == "__main__":
    main()
