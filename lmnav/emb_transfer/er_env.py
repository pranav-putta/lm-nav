import itertools
from typing import Optional

import habitat
import habitat_sim
import math
import numpy as np
from habitat import Config, Dataset
from habitat.core.dataset import EpisodeIterator
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import NavRLEnv

from lmnav.emb_transfer.utils.embodiment_navmesh_util import set_pointnav_sim_agent, set_imagenav_sim_agent


@baseline_registry.register_env(name="EmbodimentTransferRLEnv")
class EmbodimentTransferRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._core_env_config = config
        self.reward_range = self.get_reward_range()
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE
        self._previous_measure: Optional[float] = None

        # this is hacky but want to sort episodes
        # self.episodes.sort(key=lambda eps: (eps.scene_id, eps.agent_config.agent_id))
        # self._env.episode_iterator.episodes = iter(list(
        #     sorted(self._env.episode_iterator.episodes, key=lambda eps: (eps.scene_id, eps.agent_config.agent_id))))

    def compute_geodesic(self, episode):
        path = habitat_sim.ShortestPath()
        path.requested_start = episode.start_position
        path.requested_end = episode.goals[0].position

        self._env.sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        if math.isnan(reward):
            print(f'found nan >>> {self.current_episode}')
        return observations, reward, done, info

    def reset(self):
        # peek the next episode
        for episode in self._env._episode_iterator:
            set_pointnav_sim_agent(self._env._sim, episode, self._core_env_config.TASK.AGENT_CONFIGS_FILE)
            geodesic = self.compute_geodesic(episode)
            if not (math.isinf(geodesic) or math.isnan(geodesic) or geodesic <= 0):
                self._env._episode_iterator = itertools.chain([episode], self._env._episode_iterator)
                break
        return super().reset()


@baseline_registry.register_env(name="BlindEmbodimentTransferRLEnv")
class BlindEmbodimentTransferRLEnv(NavRLEnv):
    """
    Environment used for blind agents. Uses same architecture as vision agent but zeroes out the
    depth sensor observation values. 
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._core_env_config = config
        self.reward_range = self.get_reward_range()
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE
        self._previous_measure: Optional[float] = None

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        if math.isnan(reward):
            print('>>> nan episode: ', self.current_episode.scene_id, self.current_episode.episode_id,
                  self.current_episode.agent_config)
            # return observations, 0, done, info
        # add noise to observations for depth sensor
        observations['depth'] *= 0


@baseline_registry.register_env(name="NoisyCameraEmbodimentTransferRLEnv")
class NoisyCameraEmbodimentTransferRLEnv(EmbodimentTransferRLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._core_env_config = config
        self.reward_range = self.get_reward_range()
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE
        self._previous_measure: Optional[float] = None

        # self._env.sim.agents[0].agent_config.sensor_specifications[0].noise_model = 'GaussianNoiseModel'
        # self._env.sim.agents[0].agent_config.sensor_specifications[0].noise_model_kwargs = dict()

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        observations['depth'] += np.array(np.random.normal(0, 1, observations['depth'].shape),
                                          dtype=observations['depth'].dtype)
        return observations, reward, done, info


@baseline_registry.register_env(name="NoisyGPSEmbodimentTransferRLEnv")
class NoisyGPSEmbodimentTransferRLEnv(EmbodimentTransferRLEnv):
    """
    Environment used for blind agents. Uses same architecture as vision agent but zeroes out the
    depth sensor observation values.
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._core_env_config = config
        self.reward_range = self.get_reward_range()
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE
        self._previous_measure: Optional[float] = None

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        observations['pointgoal_with_gps_compass'] += np.random.normal(0, 1,
                                                                       size=observations[
                                                                           'pointgoal_with_gps_compass'].shape)
        return observations, reward, done, info


@baseline_registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.TASK.REWARD_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()


@baseline_registry.register_env(name="ImagenavEmbodimentTransferRLEnv")
class ImagenavEmbodimentTransferRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._core_env_config = config
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE
        self._previous_measure: Optional[float] = None
        self.number_episodes_skipped = 0

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.TASK.REWARD_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def compute_geodesic(self, episode):
        path = habitat_sim.ShortestPath()
        path.requested_start = episode.start_position
        path.requested_end = episode.goals[0].position

        self._env.sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)
        if math.isnan(reward):
            print(f'found nan >>> {self.current_episode}')
        return observations, reward, done, info

    def reset(self):
        # peek the next episode
        _launched_id = None
        for episode in self._env._episode_iterator._iterator:
            set_imagenav_sim_agent(self._env._sim, episode, self._core_env_config.TASK.AGENT_CONFIGS_FILE)
            geodesic = self.compute_geodesic(episode)
            if not (math.isinf(geodesic) or math.isnan(geodesic) or geodesic <= 0):
                self._env._episode_iterator._iterator = itertools.chain([episode],
                                                                        self._env._episode_iterator._iterator)
                _launched_id = episode.episode_id
                break
            self.number_episodes_skipped += 1

        self._env._episode_from_iter_on_reset = True
        if self.number_episodes_skipped % 500 == 499:
            print('number episodes skipped: ', self.number_episodes_skipped)

        out = super().reset()
        return out
