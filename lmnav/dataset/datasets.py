import os
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from lmnav.dataset.transforms import BaseDataTransform
import pickle


class BaseDataset(Dataset):
    def __init__(self, transforms: BaseDataTransform) -> None:
        self.transforms = transforms


class OfflineEpisodeDataset(BaseDataset):
    def __init__(self, transforms: BaseDataTransform, data_path=None, files=None, **kwargs):
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

        self.files = [os.path.join("data/", f.split("data/")[1]) for f in files]
        self.buffer = []
        self.file_queue = OrderedDict()
        self.default_prompt = "<s>You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
                       You can choose to move { left, right, forward, stop } at every step. The goal image is <goal>. \
                       After every image, choose the best action."



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if file.endswith(".pth") or file.endswith(".pt"):
            episode = torch.load(file)
        elif file.endswith(".pkl"):
            with open(file, "rb") as f:
                episode = pickle.load(f)
        else:
            raise NotImplementedError(f"File format {file} not recognized.")

        episode = self.transforms(episode)

        return {
            'rgb': episode['rgbs'],
            'imagegoal': episode['imagegoal'],
            'action': torch.tensor(episode['actions']),
            'prompt': self.default_prompt,
        }

class OfflineInstructionEpisodeDataset(BaseDataset):
    def __init__(self, transforms: BaseDataTransform, data_path=None, files=None, **kwargs):
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

        self.files = [os.path.join("data/", f.split("data/")[1]) for f in files]
        self.buffer = []
        self.file_queue = OrderedDict()
        self.default_prompt = "<s>You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
                       You can choose to move { left, right, forward, stop } at every step. The goal image is <goal>. \
                       After every image, choose the best action."


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if file.endswith(".pth") or file.endswith(".pt"):
            episode = torch.load(file)
        elif file.endswith(".pkl"):
            with open(file, "rb") as f:
                episode = pickle.load(f)
        else:
            raise NotImplementedError(f"File format {file} not recognized.")

        episode = self.transforms(episode)
        room_labels = " - ".join(episode['room_labels'])
        prompt = f"{self.default_prompt} The user has instructed you prefer the following path: {room_labels}."

        return {
            'rgb': episode['rgbs'],
            'imagegoal': episode['imagegoal'],
            'action': torch.tensor(episode['actions']),
            'prompt': prompt
        }




