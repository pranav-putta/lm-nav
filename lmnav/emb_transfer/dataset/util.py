import contextlib
import copy
import gzip
import itertools
import json
import os
from typing import Sequence, List, Union, Tuple

import habitat_sim
import math
import numpy as np
from habitat.core.dataset import Episode
from mltoolkit import argclass, WandBArguments
from dataclasses import field
import wandb
import attr


# STRUCTS

@argclass
class MixAndMatchAgentConfiguration:
    heights: List[float] = field(default_factory=lambda: [0.5, 1, 1.5])
    radii: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    step_sizes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    turn_incs: List[float] = field(default_factory=lambda: [10, 30, 45, 60])
    camera_fov: List[float] = field(default_factory=lambda: [90])
    depth_fov: List[float] = field(default_factory=lambda: [55])
    camera_tilt: List[float] = field(default_factory=lambda: [15])


@attr.s(auto_attribs=True)
class AgentConfiguration:
    height: float = 1.5
    radius: float = 0.1
    step_size: float = 0.25
    turn_inc: float = 10
    camera_fov: float = 90
    depth_fov: float = 90
    camera_tilt: float = 15
    agent_id: int = None

    def __eq__(self, other):
        return self.height == other.height and self.radius == other.radius \
               and self.step_size == other.step_size and self.turn_inc == other.turn_inc and \
               self.camera_fov == other.camera_fov and self.camera_tilt == other.camera_tilt


# UTILITY FUNCTIONS
def init_wandb(wandb_args: WandBArguments, config):
    """
    initializes wandb with specified arugments
    Args:
        wandb_args: WandBArguments
        config: Config to store

    Returns: returns wandb run

    """
    return wandb.init(
        name=wandb_args.run_name,
        job_type=wandb_args.job_type,
        project=wandb_args.project,
        group=wandb_args.group,
        tags=wandb_args.tags,
        config=config
    )


@contextlib.contextmanager
def allocate_device(_mp_gpu_usage, max_workers_per_gpu):
    # find device to run on
    device = None
    with _mp_gpu_usage.get_lock():
        # print(f'attempting to allocate device: {_mp_gpu_usage}')
        for gpu, worker_count in enumerate(_mp_gpu_usage):
            if worker_count < max_workers_per_gpu:
                device = gpu
                _mp_gpu_usage[gpu] += 1
                break
    yield device
    with _mp_gpu_usage.get_lock():
        _mp_gpu_usage[device] -= 1


def euclidean_dist(a, b):
    """
    computes L2 norm between two vectors
    Args:
        a: vector 1
        b: vector 2

    Returns: L2 norm

    """
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)


def scene_path_root(dataset_name, version):
    return os.path.join(f'data/scene_datasets', dataset_name, version)


def navmesh_path_from_scene(scene_path, agent_id):
    """
    computes the navmesh path given scene path and agent id
    Args:
        scene_path:
        agent_id:

    Returns:

    """
    return f'{scene_path.split(".glb")[0]}_{agent_id}.navmesh'


def dataset_path_root(name, version, split):
    return os.path.join('data/datasets', name, version, split, 'content/')


def dataset_path_from_scene(scene_path, dataset_name, version, split):
    dataset_id = os.path.dirname(scene_path).split(os.sep)[-1]
    dataset_path = os.path.join('data/datasets/', dataset_name,
                                version, f'{split}/content/', f'{dataset_id}.json.gz')
    return dataset_path


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_navmesh(sim: habitat_sim.Simulator, scene_path, agent_id):
    agent = sim.get_agent(agent_id)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = agent.agent_config.radius
    navmesh_settings.agent_height = agent.agent_config.height

    # load navmesh settings
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    # save navmesh to scene directory
    path = navmesh_path_from_scene(scene_path, agent_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sim.pathfinder.save_nav_mesh(path)


def find_action(agent, name):
    candidates = list(
        filter(
            lambda kv: kv[1].name == name,
            agent.agent_config.action_space.items(),
        )
    )
    assert len(candidates) == 1, f'could not find agent action {name}'
    return candidates[0]


def get_episodes(dataset_path):
    with gzip.open(dataset_path, 'rb') as f:
        json_str = f.readlines()[0].decode('utf-8')
        episodes = json.loads(json_str)['episodes']
    return episodes


def get_pure_pursuit_path(sim: habitat_sim.Simulator,
                          reference_path: List[Sequence[float]],
                          start_position: Sequence[float],
                          start_rotation: Sequence[float],
                          agent_config: dict,
                          scene_path: str):
    agent_config = AgentConfiguration(**agent_config)
    agent_id = agent_config.agent_id
    agent = sim.get_agent(agent_id)
    navmesh_path = navmesh_path_from_scene(scene_path, agent_id)

    sim.pathfinder.load_nav_mesh(navmesh_path)

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(start_position)
    agent_state.rotation = np.array(start_rotation)
    agent.set_state(agent_state)

    left_move, _ = find_action(agent, 'turn_left')
    right_move, _ = find_action(agent, 'turn_right')
    forward_move, _ = find_action(agent, 'move_forward')

    def next_state(last_state, ref_idx):
        # check all rotations[n] + forward
        num_turns = math.ceil(90 / agent_config.turn_inc)

        states = []
        for N in range(num_turns):
            for i in range(N):
                agent.act(left_move)
            did_collide = agent.act(forward_move)

            if did_collide:
                continue
            # compute closest point on reference path
            min_dist, min_ref = float('inf'), ref_idx
            for i in range(ref_idx, len(reference_path) - 1):
                a, b = np.array(reference_path[i]), np.array(reference_path[i + 1])
                c = np.array(agent.get_state().position)

                v_ab = b - a
                v_ac = c - a

                proj_ac_ab = v_ab * np.dot(v_ab, v_ac) / np.dot(v_ab, v_ab)
                closest_point = proj_ac_ab + a

                # check if closest_point is within the bounds of a and b
                if np.alltrue((a <= closest_point) & (closest_point <= b)):
                    dist = euclidean_dist(closest_point, c)
                else:
                    dist = min(euclidean_dist(c, a), euclidean_dist(c, b))

                if dist < min_dist:
                    min_dist = dist
                    min_ref = i

            states.append((min_dist, min_ref, agent.get_state()))
            agent.set_state(last_state)
        _, best_ref, best_state = min(states)
        return best_ref, best_state

    positions = [agent.get_state().position]
    current_ref_idx = 0
    last_state = agent.get_state()
    while current_ref_idx < len(reference_path):
        current_ref_idx, last_state = next_state(last_state, current_ref_idx)
        positions.append(last_state.position)

    return positions


def get_geodesic_path_episode(sim: habitat_sim.Simulator,
                              episode: Episode,
                              success_dist: float,
                              agent_id: int,
                              scene_path: str):
    return get_geodesic_path(sim, episode.start_position, episode.start_rotation, episode.goals[0]['position'],
                             success_dist, agent_id, scene_path)


def get_geodesic_path(sim: habitat_sim.Simulator,
                      start_position: Sequence[float],
                      start_rotation: Sequence[float],
                      end_position: Sequence[float],
                      success_dist: float,
                      agent_id: int,
                      scene_path: str,
                      return_rotations=False):
    follower = sim.make_greedy_follower(agent_id, goal_radius=success_dist, forward_key='move_forward',
                                        left_key='turn_left', right_key='turn_right')
    agent = sim.get_agent(agent_id)

    navmesh_path = navmesh_path_from_scene(scene_path, agent_id)
    sim.pathfinder.load_nav_mesh(navmesh_path)

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(start_position)
    agent_state.rotation = np.array(start_rotation)
    agent.set_state(agent_state)
    positions = [agent.get_state().position]
    rotations = [agent.get_state().rotation]
    try:
        actions = follower.find_path(end_position)
    except:
        if return_rotations:
            return [], []
        return []

    for action in actions:
        if action is None:
            continue
        agent.act(action)
        state = agent.get_state()
        if 'turn' not in action:
            positions.append(state.position)
            rotations.append(state.rotation)
    if return_rotations:
        return positions, rotations
    return positions


def compute_path_dist(path: Sequence[Sequence[float]]):
    """
    computes distance of path by computing the euclidean distance of each segment
    Args:
        path: list of points

    Returns: path length

    """
    dist = 0
    last = path[0]
    for point in path[1:]:
        dist += euclidean_dist(last, point)
        last = point
    return dist


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.
    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def create_agent(config: AgentConfiguration, camera_spec=None):
    def create_camera_spec():
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f'color_sensor'
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [1280, 960]
        rgb_sensor_spec.position = [0., 1.5, 0.]
        rgb_sensor_spec.orientation = [0., 0., 0.]
        return rgb_sensor_spec

    if camera_spec is None:
        camera_spec = create_camera_spec()
    else:
        tmp = create_camera_spec()
        tmp.channels = camera_spec.channels
        tmp.clear_color = camera_spec.clear_color
        tmp.far = camera_spec.far
        tmp.near = camera_spec.near
        tmp.sensor_subtype = camera_spec.sensor_subtype
        tmp.sensor_type = camera_spec.sensor_type
        tmp.uuid = camera_spec.uuid
        tmp.resolution = camera_spec.resolution
        camera_spec = tmp

    agent = habitat_sim.agent.AgentConfiguration()

    # modify agent height, radius, stepsize and turninc
    agent.height = config.height
    agent.radius = config.radius
    agent.action_space[1].actuation.amount = config.step_size
    agent.action_space[2].actuation.amount = config.turn_inc
    agent.action_space[3].actuation.amount = config.turn_inc

    # modify camera spec for hfov and orientation
    camera_spec.hfov = config.camera_fov
    camera_spec.orientation = [np.deg2rad(config.camera_tilt), 0, 0]
    camera_spec.position = [0, config.height, 0]

    return agent


def init_sim(scene_path: str, agent_configs: List[AgentConfiguration], device_id=0):
    """
    initializes simulator
    Args:
        scene_path: path to ".glb" scene path
        agent_configs: list of agent configurations
        device_id: gpu device to use

    Returns:

    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = device_id

    def create_camera_spec():
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = f'color_sensor'
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [1280, 960]
        rgb_sensor_spec.position = [0., 1.5, 0.]
        rgb_sensor_spec.orientation = [0., 0., 0.]
        return rgb_sensor_spec

    # initialize agent configurations
    agents = []
    for agent_cfg in agent_configs:
        agent = habitat_sim.agent.AgentConfiguration()
        agent.radius = agent_cfg.radius
        agent.height = agent_cfg.height
        agent.action_space['move_forward'].actuation.amount = agent_cfg.step_size
        agent.action_space['turn_left'].actuation.amount = agent_cfg.turn_inc
        agent.action_space['turn_right'].actuation.amount = agent_cfg.turn_inc

        # include other agent configs in here
        agent.sensor_specifications = [create_camera_spec()]
        agents.append(agent)
    cfg = habitat_sim.Configuration(sim_cfg, agents)
    sim = habitat_sim.Simulator(cfg)
    sim.reset()

    return sim


def compute_agent_combos(args: MixAndMatchAgentConfiguration):
    agent_combos_requires_navmesh = list(
        itertools.product(
            *[args.heights, args.radii, args.step_sizes, args.turn_incs]))
    agent_combos_camera_configs = list(itertools.product(*[args.camera_fov, args.camera_tilt]))
    agent_combos_camera_configs = [
        (*x, args.depth_fov[int(i // (len(agent_combos_camera_configs) / len(args.depth_fov)))]) for i, x in
        enumerate(agent_combos_camera_configs)]

    return [AgentConfiguration(height=config[0], radius=config[1], step_size=config[2], turn_inc=config[3],
                               camera_fov=camera_config[0], depth_fov=camera_config[2], camera_tilt=camera_config[1],
                               agent_id=_id)
            for camera_config in agent_combos_camera_configs
            for _id, config in enumerate(agent_combos_requires_navmesh)]
