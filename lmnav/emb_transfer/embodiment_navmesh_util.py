import os
import numpy as np


def get_navmesh_name(episode, agent_configs_file):
    scene_id = episode.scene_id.split(os.sep)
    root_path, scene_path = os.sep.join(scene_id[:-2]), os.sep.join(scene_id[-2:])
    ac_file_dir = os.path.splitext(os.path.basename(agent_configs_file))[0]
    scene_path = os.path.join(root_path, ac_file_dir, scene_path)
    return os.path.splitext(scene_path)[0] + f'_{episode.agent_config.agent_id}.navmesh'


def set_imagenav_sim_agent(sim, episode, agent_configs_file, old_navmesh_filename=''):
    navmesh_filename = get_navmesh_name(episode, agent_configs_file)

    if old_navmesh_filename != navmesh_filename:
        if not sim.pathfinder.load_nav_mesh(navmesh_filename):
            raise Exception(f'could not load the navmesh {navmesh_filename}')
    # change the height, radius, step_size and turn angle
    sim.agents[0].agent_config.height = episode.agent_config.height
    sim.agents[0].agent_config.radius = episode.agent_config.radius
    sim.agents[0].agent_config.action_space[1].actuation.amount = episode.agent_config.step_size
    sim.agents[0].agent_config.action_space[2].actuation.amount = episode.agent_config.turn_inc
    sim.agents[0].agent_config.action_space[3].actuation.amount = episode.agent_config.turn_inc

    sim.agents[0].agent_config.sensor_specifications[0].hfov = episode.agent_config.camera_fov
    sim.agents[0].agent_config.sensor_specifications[0].orientation = [
        np.deg2rad(episode.agent_config.camera_tilt), 0, 0]
    sim.agents[0].agent_config.sensor_specifications[0].position = [0, episode.agent_config.height, 0]
    sim.agents[0]._sensors['rgb'].fov = episode.agent_config.camera_fov

    if 'depth' in sim.agents[0]._sensors:
        sim.agents[0]._sensors['depth'].fov = episode.agent_config.depth_fov
        sim.agents[0].agent_config.sensor_specifications[1].hfov = episode.agent_config.depth_fov
        sim.agents[0].agent_config.sensor_specifications[1].orientation = [
            np.deg2rad(episode.agent_config.camera_tilt), 0, 0]
        sim.agents[0].agent_config.sensor_specifications[1].position = [0, episode.agent_config.height, 0]

    sim.set_agent_state(episode.start_position, episode.start_rotation)


def set_pointnav_sim_agent(sim, episode, agent_configs_file, old_navmesh_filename=''):
    navmesh_filename = get_navmesh_name(episode, agent_configs_file)

    if old_navmesh_filename != navmesh_filename:
        if not sim.pathfinder.load_nav_mesh(navmesh_filename):
            raise Exception(f'could not load the navmesh {navmesh_filename}')
    # change the height, radius, step_size and turn angle
    sim.agents[0].agent_config.height = episode.agent_config.height
    sim.agents[0].agent_config.radius = episode.agent_config.radius
    sim.agents[0].agent_config.action_space[1].actuation.amount = episode.agent_config.step_size
    sim.agents[0].agent_config.action_space[2].actuation.amount = episode.agent_config.turn_inc
    sim.agents[0].agent_config.action_space[3].actuation.amount = episode.agent_config.turn_inc

    sim.agents[0].agent_config.sensor_specifications[0].hfov = episode.agent_config.camera_fov
    sim.agents[0].agent_config.sensor_specifications[0].orientation = [
        np.deg2rad(episode.agent_config.camera_tilt), 0, 0]
    sim.agents[0].agent_config.sensor_specifications[0].position = [0, episode.agent_config.height, 0]
    sim.agents[0]._sensors['depth'].fov = episode.agent_config.camera_fov

    sim.set_agent_state(episode.start_position, episode.start_rotation)