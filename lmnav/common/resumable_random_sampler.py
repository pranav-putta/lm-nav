import torch
import math

class DistributedResumableSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source, rank, world_size, batch_size):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator = self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        
        # drop the tail to make sure data source is evenly divisible
        self.perm = torch.randperm(len(self.data_source), generator=self.generator)
        self.total_size = self.num_samples * self.world_size
        
    @property
    def num_samples(self) -> int:
        # drop the tail to make sure data source is evenly divisible
        block_size = self.world_size * self.batch_size
        return math.floor((len(self.data_source) - block_size) / block_size) * self.batch_size
    
    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        indices = self.perm[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
    def state_dict(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def load_state_dict(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])
