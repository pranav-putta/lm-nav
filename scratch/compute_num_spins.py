import torch
import os
from tqdm import tqdm


def find_subsequences_of_2s(nums_tensor):
    # Find starting and ending positions of 2s
    starts = (nums_tensor[:-1] != 2) & (nums_tensor[1:] == 2)
    ends = (nums_tensor[:-1] == 2) & (nums_tensor[1:] != 2)

    # Add position for the case when the tensor starts or ends with a 2
    starts = torch.cat([torch.tensor([nums_tensor[0] == 2]), starts])
    ends = torch.cat([ends, torch.tensor([nums_tensor[-1] == 2])])

    # Extract the indices
    start_indices = torch.nonzero(starts).squeeze(dim=-1).tolist()
    end_indices = torch.nonzero(ends).squeeze(dim=-1).tolist()

    indices = list(zip(start_indices, end_indices))
    return indices


path = "data/datasets/lmnav/offline_00744"
files = os.listdir(path)

total_uturns = 0
total = 0

pbar = tqdm(files)
for file in pbar:
    data = torch.load(os.path.join(path, file))

    subseqs = find_subsequences_of_2s(data["action"])
    numuturns = sum((e - s) >= 11 for s, e in subseqs)
    total_uturns += numuturns
    total += 1

    pbar.set_description(f"{total_uturns} / {total}")
