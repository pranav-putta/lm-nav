import os
from threading import Thread
import gzip
import json
import sys
import copy
import habitat
import habitat.config
import multiprocessing as mp
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from habitat.utils.visualizations.maps import calculate_meters_per_pixel 
from tqdm import tqdm
import torch
import pickle

from lmnav.config.default import get_config
from lmnav.dataset.offline_trajectory_dataset import *

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def get_room_of_pixel(coords, room_data, pathfinder):
    """
    Get the room name in which the given pixel is located.

    Args:
    pixel_coords (tuple): The coordinates of the pixel.
    rooms (dict): A dictionary of rooms with their coordinates.

    Returns:
    str: The name of the room in which the pixel is located, or 'None' if the pixel is not in any room.
    """


    # get room level
    levels, rooms = room_data['levels'], room_data['rooms']

    # get the room level that the pixel is in
    room_level = None
    for start, end in levels:
        if start <= coords[1] < end:
            room_level = end
            break

    if room_level is None:
        raise Exception(f"Pixel {coords} is not in any room level")

    # convert pixel coordinates to topdown coordinates
    meters_per_pixel = calculate_meters_per_pixel(128, pathfinder=pathfinder)
    pixel_coords = convert_points_to_topdown(pathfinder, [coords], meters_per_pixel)[0]


    room_list = []
    for room, bbox in rooms[room_level].items():
        (x_top, y_top), (x_bottom, y_bottom) = bbox
        # pixel coords axis is flipped
        if x_top <= pixel_coords[1] < x_bottom and y_top <= pixel_coords[0] < y_bottom:
            room_list.append(room)

    if len(room_list) == 0:
        # print(f"A pixel didn't get any room! ({pixel_coords})")
        return None
    if len(room_list) > 1:
        # print(f"A pixel belongs to more than one room, which should not happen. ({pixel_coords})")
        return None

    return room_list[0] if room_list else None



def main():
    os.chdir("/srv/flash1/pputta7/projects/lm-nav")

    cfg_path = sys.argv[1]
    room_path = sys.argv[2]

    config = get_config(cfg_path)

    new_path = os.path.join(config.generator.store_artifact.dirpath, f"{config.generator.store_artifact.name}+clip")
    files = os.listdir(new_path)
    print(f"Loading dataset from {new_path}")

    # Load the room labels
    with open(room_path, 'r') as f:
        room_data = json.load(f)

    with habitat.Env(config=config.habitat) as env:
        for episode in tqdm(files):
            episode_path = os.path.join(new_path, episode)
            episode = torch.load(episode_path)
            
            # Get the room labels for each episode
            room_labels_episode = []
            for coord in episode['coords']:
                room = get_room_of_pixel(coord, room_data, env.sim.pathfinder)
                if room is not None:
                    room_labels_episode.append(room)
                else:
                    # print(f"Episode {episode.episode_id} has pixel {coord} not in any room")
                    pass

            # remove consecutive duplicates
            room_labels_episode = [room for i, room in enumerate(room_labels_episode) if i == 0 or room != room_labels_episode[i-1]]
            
            episode['room_labels'] = room_labels_episode

            # Save the episode back to the file
            torch.save(episode, episode_path)

if __name__ == "__main__":
    main()
