import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch
import habitat

from lmnav.data_gen import _init_envs



class OfflineTrajectoryDataset(Dataset):
    """
    This dataset is different than offline episode dataset. 
    It assumes that only the action data is available in cache and generates
    the rest in simulation
    """

    def __init__(self, data_path=None, files=None):
        assert (data_path is None) ^ (files is None), "only one of 'data_path' and 'files' is expected"
        
        # set metadata using the config file
        if data_path is not None:
            self.files = [os.path.join(data_path, file) for file in sorted(os.listdir(data_path))]
        else:
            self.files = list(sorted(files))
        self.buffer = []
        self.file_queue = OrderedDict()
    

    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, idx):
        file = self.files[idx]
        episode = torch.load(file)
        return episode
