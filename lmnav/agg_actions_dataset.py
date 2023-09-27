import torch
import os
from tqdm import tqdm
from multiprocessing import Pool

datadir = "/srv/flash1/pputta7/projects/lm-nav/data/datasets/lmnav/offline_10envs/"
files = list(sorted([os.path.join(datadir, file) for file in sorted(os.listdir(datadir))]))


def aggregate_data(idx):
    try:
        data = torch.load(files[idx])
        return (idx, data['action'])
    except:
        print("there was a problem at file", idx)
        return None

num_workers = 16
partition_size = 50
results = []
for i in range(partition_size):
    with Pool(num_workers) as p: 
        idxs = range(i, len(files), partition_size)
        partition_result = list(tqdm(p.imap(aggregate_data, idxs), total=len(idxs), desc=f'Partition {i}'))
        partition_result = [p for p in partition_result if p is not None]
        results += partition_result
        torch.save(results, 'action_dataset_2.pt') 
