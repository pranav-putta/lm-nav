import os
import copy
import queue
import sys
import habitat
import numpy as np
import pickle
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
import torch
import multiprocessing as mp

from torch.multiprocessing import Process
from lmnav.common.utils import convert_weights_to_fp16
from lmnav.config.default import get_config
from tqdm import tqdm
from torchvision import transforms

from lmnav.dataset.offline_trajectory_dataset import OfflineTrajectory
from transformers import CLIPProcessor, CLIPModel

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet" 

device = "cuda:0"

def _extract_episode_data_from_trajectory(file_path):
    """
    Process a single file to extract scene_id and episode_id.
    """
    try:
        if file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.endswith(".pt"):
            data = torch.load(file_path)
        else:
            raise Exception(f"Unknown file format: {file_path}")
        return data.get('action'), data.get("scene_id"), data.get("episode_id"), file_path.split(".")[1]
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {e}")


def construct_dataset(config):
    NUM_PROCESSES = 32

    print("Step 1: Merge trajectory dataset")
    print("-"*80)
    print("Constructing merged offline dataset...")


    # config variables
    generator_cfg = config.generator
    artifact_dir = os.path.join(generator_cfg.store_artifact.dirpath, generator_cfg.store_artifact.name)

    # load original dataset used to collect trajectory
    dataset = habitat.make_dataset(
            id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    episodes_dict = {(episode.scene_id, episode.episode_id): episode for episode in dataset.episodes}
    episodes = []
    print(f"Loaded original dataset from {config.habitat.dataset.data_path}")

    # record actions and trajectory id for each trajectory in dataset
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Create a list of file paths to process
        filter_file_fn = lambda file: file.startswith("data") and (file.endswith(".pt") or file.endswith(".pkl"))
        file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(artifact_dir) for file in filter(filter_file_fn, files)]
        total_files = len(file_paths)

        # get new file location
        new_path = os.path.join(config.generator.store_artifact.dirpath, f"{config.generator.store_artifact.name}+clip_new")
        os.makedirs(new_path, exist_ok=True)

        # remove files in file_paths that already exist in new_path
        processed_trajectories = set([os.path.basename(f).split(".")[1] for f in os.listdir(new_path)])
        file_paths = [f for f in file_paths if not os.path.basename(f).split(".")[1] in processed_trajectories]
        print(f"Skipping {total_files - len(file_paths)} files that already exist in {new_path}")
        total_files = len(file_paths)

        # file_paths = file_paths[:1000]
        print(f"Loading {len(file_paths)} trajectories from {artifact_dir}")
               
        # Process files in parallel with a progress bar
        for actions, scene_id, episode_id, trajectory_id in tqdm(pool.imap_unordered(_extract_episode_data_from_trajectory, file_paths), total=len(file_paths)):
            base_episode = episodes_dict[(scene_id, episode_id)]
            episodes.append(OfflineTrajectory.from_nav_episode(base_episode, actions.tolist(), trajectory_id))

    # return dataset
    new_dataset = habitat.Dataset()
    new_dataset.episodes = episodes
    
    assert len(new_dataset.episodes) == len(episodes), "Dataset length mismatch after construction"
    print(f"Constructed dataset with {len(new_dataset.episodes)} trajectories")
    return new_dataset

def _process_dataset(config, dataset, start, end, q):
    dataset.episodes = dataset.episodes[start:end]
    print(f"Processing episodes {start} to {end}; total dataset size {len(dataset.episodes)}")
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        # each episode has a list of actions
        # run each episode through the simulator and collect the agent's coordinates
        for i in range(len(dataset.episodes)):
            obs = env.reset()
            episode = env.current_episode

            # Loop through episode actions and collect resampled data
            coords, rgbs, imagegoal = [], [], obs["imagegoal"]
            success = False
            for action in episode.actions:
                coords.append(
                    env.sim.get_agent_state().position.tolist(),
                )
                rgbs.append(
                    obs["rgb"].copy()
                )
                metrics = env.get_metrics()
                success = metrics["distance_to_goal"] < 1.0

                obs = env.step(action)

            assert success, "Episode failed to reach goal"
            trajectory = {
                "coords": coords,
                "rgbs": rgbs,
                "actions": episode.actions,
                "imagegoal": imagegoal,
                "scene_id": episode.scene_id,
                "episode_id": episode.episode_id,
                "trajectory_id": episode.trajectory_id
            }

            # push trajectory to queue
            q.put(trajectory)

def _embed_trajectories(trajectories, model, processor):
    episode_lengths = [len(trajectory['rgbs']) for trajectory in trajectories]
    rgb_frames = np.concatenate([np.stack(trajectory['rgbs']) for trajectory in trajectories], axis=0)
    goal_frames = np.stack([trajectory['imagegoal'] for trajectory in trajectories])
    
    x = np.concatenate([rgb_frames, goal_frames], axis=0)
    x = torch.from_numpy(x)
    # x = processor(images=x, return_tensors="pt", device=device)
    x = processor(x.permute(0, 3, 1, 2))
    x = x.to(device)
    print(f"Embedding {x.shape} frames")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        x = model.vision_model(x)
    embeddings = x.pooler_output.detach().cpu()

    rgb_embds, goal_embds = torch.split(embeddings, [len(rgb_frames), len(goal_frames)])
    rgb_embds = torch.split(rgb_embds, episode_lengths)
    for i in range(len(trajectories)):
        trajectories[i]['rgbs'] = rgb_embds[i]
        trajectories[i]['imagegoal'] = goal_embds[i]
    return trajectories

def resample_dataset(config, dataset):
    NUM_PROCESSES = 16

    print("Step 2: Resample dataset")
    print("-" * 80)
    
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
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
            }
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 224 
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 224 

    dataset_split_size = 1 + (len(dataset.episodes) // NUM_PROCESSES)
    print(f"Splitting dataset into {NUM_PROCESSES} chunks of size {dataset_split_size}")

    # Process files in parallel with a progress bar
    mp_ctx = mp.get_context("forkserver")
    mp_queue = mp_ctx.Queue()
    processes = []
    total_processed = 0
    for i in range(min(NUM_PROCESSES, len(dataset.episodes))):
        start = i * dataset_split_size
        end = min((i + 1) * dataset_split_size, len(dataset.episodes))
        p = mp_ctx.Process(
            target=_process_dataset,
            args=(
                config,
                copy.deepcopy(dataset),
                start,
                end,
                mp_queue
            )
        )
        p.start()
        processes.append(p)

    # Collect results from the workers through the pool result queue
    # then store them into a new trajectory dataset
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)
    model = model.eval()
    convert_weights_to_fp16(model)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    processor = transforms.Compose(
            [
                transforms.Resize(
                    224,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                    inplace=True,
                ),
            ]
        )


    buffer = []
    new_path = os.path.join(config.generator.store_artifact.dirpath, f"{config.generator.store_artifact.name}+clip_new")
    os.makedirs(new_path, exist_ok=True)
    print(f"Saving new dataset to {new_path}")

    with tqdm(total=len(dataset.episodes), desc="Resampling dataset") as pbar:
        while total_processed < len(dataset.episodes):
            buffer.append(mp_queue.get())
            total_processed += 1
            pbar.update(1)

            num_frames = sum([len(trajectory['rgbs']) for trajectory in buffer])

            if num_frames >= 900 or total_processed == len(dataset.episodes):
                embedded_trajectories = _embed_trajectories(buffer, model, processor)
                for trajectory in embedded_trajectories:
                    torch.save(trajectory, os.path.join(new_path, f"data.{trajectory['trajectory_id']}.pt"))
                buffer = []

    print("Finished processing dataset!")
    # Wait for all worker processes to finish
    for p in processes:
        p.join()

def main():
    os.chdir("/srv/flash1/pputta7/projects/lm-nav")
    
    cfg_path = sys.argv[1]
    config = get_config(cfg_path)

    # 1. Construct dataset from existing archive
    dataset = construct_dataset(config)

    # 2. Resample data with new camera resolution and store agent coordinates
    resample_dataset(config, dataset)

if __name__ == "__main__":
    main()
