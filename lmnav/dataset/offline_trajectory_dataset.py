# [setup]
import os
from typing import TYPE_CHECKING, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.core.registry import registry
import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

class OfflineTrajectory(NavigationEpisode):

    def __init__(self, episode_id, scene_id, scene_dataset_config, additional_obj_config_paths, start_position, start_rotation, info, goals, start_room, shortest_paths, actions, trajectory_id, agent_coords=None, room_labels=None):
        # Initialize the base NavigationEpisode class with attributes from the base episode
        super().__init__(
            episode_id=episode_id,
            scene_id=scene_id,
            scene_dataset_config=scene_dataset_config,
            additional_obj_config_paths=additional_obj_config_paths,
            start_position=start_position,
            start_rotation=start_rotation,
            info=info,
            goals=goals,
            start_room=start_room,
            shortest_paths=shortest_paths
        )

        # Add the actions variable to the class
        self.actions = actions
        self.trajectory_id = trajectory_id
        self.agent_coords = agent_coords
        self.room_labels = room_labels

    @staticmethod
    def from_nav_episode(base_episode, actions, trajectory_id):
        return OfflineTrajectory(
            episode_id=base_episode.episode_id,
            scene_id=base_episode.scene_id,
            scene_dataset_config=base_episode.scene_dataset_config,
            additional_obj_config_paths=base_episode.additional_obj_config_paths,
            start_position=base_episode.start_position,
            start_rotation=base_episode.start_rotation,
            info=base_episode.info,
            goals=base_episode.goals,
            start_room=base_episode.start_room,
            shortest_paths=base_episode.shortest_paths,
            actions=actions,
            trajectory_id=trajectory_id
        )



@registry.register_dataset(name="OfflineTrajectoryDataset-v1")
class OfflineTrajectoryDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
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
                cfg.content_scenes = [ALL_SCENES_MASK]
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

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """
        Load the data from a file into `self.episodes`. This can load `.pickle`
        or `.json.gz` file formats.
        """

        if fname.endswith(".pickle"):
            # NOTE: not implemented for pointnav
            with open(fname, "rb") as f:
                self.from_binary(pickle.load(f), scenes_dir=scenes_dir)
        else:
            with gzip.open(fname, "rt") as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None, directory=None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.data_path.format(split=config.split)

        self._load_from_file(datasetfile_path, config.scenes_dir)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )

                self._load_from_file(scene_filename, config.scenes_dir)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )


    def to_binary(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        raise NotImplementedError()

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = OfflineTrajectory(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)


