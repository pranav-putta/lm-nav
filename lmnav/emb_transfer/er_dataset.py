#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import random
from typing import List, Optional, Sequence

import attr
from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset, Episode, EpisodeIterator
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal, ShortestPathPoint
from mltoolkit import parse_config
import copy
import numpy as np

from tqdm import tqdm

from lmnav.emb_transfer.dataset.util import MixAndMatchAgentConfiguration, compute_agent_combos, AgentConfiguration
from lmnav.emb_transfer.er_episode import EmbodimentEpisode

# from structs import AgentConfiguration

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


class EREvalEpisodeIterator(EpisodeIterator):
    def __init__(self, episodes: Sequence, cycle: bool = True, shuffle: bool = False, group_by_scene: bool = True,
                 max_scene_repeat_episodes: int = -1, max_scene_repeat_steps: int = -1, num_episode_sample: int = -1,
                 step_repetition_range: float = 0.2, seed: int = None) -> None:
        super().__init__(episodes, cycle, shuffle, group_by_scene, max_scene_repeat_episodes, max_scene_repeat_steps,
                         num_episode_sample, step_repetition_range, seed)


@registry.register_dataset(name="ERNav-v1")
class ERDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[EmbodimentEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(config.DATA_PATH.format(split=config.SPLIT))
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(f"Could not find dataset file `{dataset_dir}`")

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = cls(cfg)
        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.CONTENT_SCENES = [ALL_SCENES_MASK]
            dataset = cls(cfg)
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
            content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(data_path=dataset_dir)
        )
        if has_individual_scene_files:
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

        print(len(self.episodes))

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = EmbodimentEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            episode.agent_config = AgentConfiguration(**episode.agent_config)
            self.episodes.append(episode)


@registry.register_dataset(name="ERNavDynamicAgentLoading-v1")
class ERNavDynamicAgentLoading(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[EmbodimentEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(config.DATA_PATH.format(split=config.SPLIT))
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(f"Could not find dataset file `{dataset_dir}` or `{config.SCENES_DIR}`")

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = cls(cfg)
        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.CONTENT_SCENES = [ALL_SCENES_MASK]
            dataset = cls(cfg)
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    def get_episode_iterator(self, *args, **kwargs):
        return EREvalEpisodeIterator(self.episodes, *args, **kwargs)

    @staticmethod
    def _get_scenes_from_folder(
            content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        # load agent configs
        agent_configs_filename = config.AGENT_CONFIGS_FILE
        agent_config_args: MixAndMatchAgentConfiguration = MixAndMatchAgentConfiguration(
            **parse_config(agent_configs_filename))

        # if only changing camera configs, only keep camera config params
        if config.ONLY_CHANGE_CAMERA_CONFIGS:
            agent_config_args.radii = [None]
            agent_config_args.turn_incs = [None]
            agent_config_args.heights = [None]
            agent_config_args.step_sizes = [None]
        self.agent_configs = compute_agent_combos(agent_config_args)

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(data_path=dataset_dir)
        )

        if has_individual_scene_files:
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.SCENES_DIR,
                                   max_episodes=config.MAX_EPISODES_PER_SCENE,
                                   max_agents_per_episode=config.MAX_AGENTS_PER_EPISODE,
                                   only_change_camera_configs=config.ONLY_CHANGE_CAMERA_CONFIGS)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None, max_episodes=-1,
                  max_agents_per_episode=-1, only_change_camera_configs=False) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        count = 0
        iter_episodes = deserialized["episodes"]
        if max_episodes >= 0:
            random.shuffle(iter_episodes)
            iter_episodes = iter_episodes[:max_episodes]

        for episode in iter_episodes:
            episode = EmbodimentEpisode(**episode)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            if max_agents_per_episode == -1:
                agent_config_samples = self.agent_configs
            else:
                agent_config_samples_indices = np.array(list(range(len(self.agent_configs))))
                np.random.shuffle(agent_config_samples_indices)
                agent_config_samples = [self.agent_configs[i] for i in
                                        agent_config_samples_indices[:max_agents_per_episode]]

            for agent_config in agent_config_samples:
                new_eps = copy.deepcopy(episode)
                if only_change_camera_configs:
                    new_eps.agent_config = AgentConfiguration(**new_eps.agent_config)
                    new_eps.agent_config.camera_fov = agent_config.camera_fov
                    new_eps.agent_config.camera_tilt = agent_config.camera_tilt
                    new_eps.agent_config.depth_fov = agent_config.depth_fov
                else:
                    new_eps.agent_config = (copy.deepcopy(agent_config))

                new_eps.episode_id = count
                self.episodes.append(new_eps)
                count += 1
