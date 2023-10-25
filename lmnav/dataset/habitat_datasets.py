from typing import Optional
import torch
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

@registry.register_dataset(name='CustomFilterDataset')
class CustomFilterDataset(PointNavDatasetV1):
    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        super().__init__(config)

        # randomly sample 250 episodes
        # use a torch generator to sample 250 episodes
        # filter out episodes with geodesic distance < 5.0

        generator = torch.Generator()
        generator.manual_seed(42)
        indicies = torch.randperm(len(self.episodes), generator=generator)
        self.episodes = [self.episodes[i] for i in indicies]
        self.episodes = list(filter(lambda episode: episode.info['geodesic_distance'] > 5.0, self.episodes))
        self.episodes = self.episodes[:500]

