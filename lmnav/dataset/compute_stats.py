import multiprocessing
import pickle
import sys
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

def process_file(file_path):
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
            raise NotImplementedError(f"File type {file_path} not supported.")
        return data.get('scene_id'), data.get('episode_id')

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def extract_ids_with_progress(directory):
    scene_ids = []
    episode_ids = []

    # Create a list of file paths to process
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(directory) 
                  for file in files if file.endswith(".pt") or file.endswith(".pkl")]
    print(len(file_paths))

    # Initialize a multiprocessing pool
    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} cpus.")
    pool = multiprocessing.Pool(processes=num_cpus)

    # Process files in parallel with a progress bar
    for scene_id, episode_id in tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths)):
        if scene_id is not None:
            scene_ids.append(scene_id)
        if episode_id is not None:
            episode_ids.append(episode_id)

    # Close the pool
    pool.close()
    pool.join()

    return scene_ids, episode_ids

# get directory from system arguments
directory = sys.argv[1]
scene_ids, episode_ids = extract_ids_with_progress(directory)

output = {
    'scene_ids': scene_ids,
    'episode_ids': episode_ids
}
with open(os.path.join(directory, 'stats.pkl'), 'wb+') as f:
    pickle.dump(output, f)
