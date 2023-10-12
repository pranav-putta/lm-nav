from typing import Any
import torch
import random


class BaseDataTransform:
    def __call__(self, x) -> Any:
        return x


class ReverseTurnsTransform(BaseDataTransform):
    def find_subsequences_of_2s(self, nums_tensor):
        # Find starting and ending positions of 2s
        starts = (nums_tensor[:-1] != 2) & (nums_tensor[1:] == 2)
        ends = (nums_tensor[:-1] == 2) & (nums_tensor[1:] != 2)

        # Add position for the case when the tensor starts or ends with a 2
        if nums_tensor[0] == 2:
            starts = torch.cat([torch.tensor([True]), starts])
        else:
            starts = torch.cat([torch.tensor([False]), starts])

        if nums_tensor[-1] == 2:
            ends = torch.cat([ends, torch.tensor([True])])
        else:
            ends = torch.cat([ends, torch.tensor([False])])

        # Extract the indices
        start_indices = torch.nonzero(starts).squeeze(dim=-1).tolist()
        end_indices = torch.nonzero(ends).squeeze(dim=-1).tolist()

        indices = list(zip(start_indices, end_indices))
        return indices

    def __call__(self, x) -> Any:
        seq_idxs = self.find_subsequences_of_2s(x["action"])

        for s, e in seq_idxs:
            length = e - s + 1
            if length >= 12:
                idx = random.randint(s, e - 11)
                slice_ = slice(idx, idx + 11, 1)
                x["rgb"][slice_] = torch.flip(x["rgb"][slice_], dims=(0,))
                x["depth"][slice_] = torch.flip(x["depth"][slice_], dims=(0,))
                x["reward"][slice_] = torch.flip(x["reward"][slice_], dims=(0,))
                x["action"][slice_] = 3

        return x
