import os
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from lmnav.dataset.transforms import BaseDataTransform


class BaseDataset(Dataset):
    def __init__(self, transforms: BaseDataTransform) -> None:
        self.transforms = transforms


class OfflineEpisodeDataset(BaseDataset):
    def __init__(self, transforms: BaseDataTransform, data_path=None, files=None):
        super().__init__(transforms)
        assert (data_path is None) ^ (
            files is None
        ), "only one of 'data_path' and 'files' is expected"

        # set metadata using the config file
        if data_path is not None:
            self.files = [
                os.path.join(data_path, file) for file in sorted(os.listdir(data_path))
            ]
        else:
            self.files = list(sorted(files))
        self.buffer = []
        self.file_queue = OrderedDict()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        episode = torch.load(file)
        episode = self.transforms(episode)
        return episode
