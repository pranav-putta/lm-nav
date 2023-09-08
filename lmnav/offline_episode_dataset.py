import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch
import habitat

class OfflineEpisodeDataset(Dataset):

    def __init__(self, data_path):
        # set metadata using the config file
        self.files = [os.path.join(data_path, file) for file in sorted(os.listdir(data_path))]
        self.buffer = []

        self.file_queue = OrderedDict()
    

    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, idx):
        file = self.files[idx]
        episode = torch.load(file)
        return episode
