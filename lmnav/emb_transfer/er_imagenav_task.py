import itertools
import os
from typing import Any, Optional, Dict, Sequence

import habitat
import habitat_sim
import numpy as np

from habitat import Config, logger
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationTask
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim import Agent

from lmnav.emb_transfer.dataset.util import create_agent
from lmnav.emb_transfer.er_dataset import EmbodimentEpisode, AgentConfiguration
from lmnav.emb_transfer.embodiment_navmesh_util import set_imagenav_sim_agent, get_navmesh_name


def merge_sim_radius_height_config(sim_config: Config, episode: EmbodimentEpisode) -> Any:
    agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
    agent_cfg = getattr(sim_config, agent_name)
    agent_cfg.defrost()
    agent_cfg.HEIGHT = episode.agent_config.height
    agent_cfg.RADIUS = episode.agent_config.radius
    agent_cfg.freeze()
    sim_config.defrost()
    sim_config.DEPTH_SENSOR.POSITION = [0.0, agent_cfg.HEIGHT, 0.0]
    sim_config.RGB_SENSOR.POSITION = [0.0, agent_cfg.HEIGHT, 0.0]
    # sim_config.navmesh_agent_id = str(0)  # episode.agent_config["agent_id"])
    sim_config.FORWARD_STEP_SIZE = episode.agent_config.step_size
    sim_config.TURN_ANGLE = episode.agent_config.turn_inc
    sim_config.freeze()
    return sim_config


@registry.register_task(name="ERImageNav-v0")
class ERImageNavTask(NavigationTask):
    def __init__(
            self, config: Config, sim: habitat.sims.habitat_simulator.habitat_simulator.HabitatSim,
            dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        # little hacky
        self.current_nav_mesh = None
        self.scene_id = None

    def overwrite_sim_config(self, sim_config: Any, episode: EmbodimentEpisode) -> Any:
        sim_config = super().overwrite_sim_config(sim_config, episode)
        return merge_sim_radius_height_config(sim_config, episode)

    def reset(self, episode: EmbodimentEpisode):
        set_imagenav_sim_agent(self._sim, episode, self._config.AGENT_CONFIGS_FILE,
                               old_navmesh_filename=self.current_nav_mesh)
        self.current_nav_mesh = get_navmesh_name(episode, self._config.AGENT_CONFIGS_FILE)
        return super().reset(episode)
