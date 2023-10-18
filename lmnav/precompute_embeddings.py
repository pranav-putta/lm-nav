import argparse
import torch
import einops
import math

from lmnav.config.default import get_config

from hydra.utils import instantiate

from lmnav.common.writer import *

from lmnav.dataset.datasets import OfflineEpisodeDataset
from lmnav.models.base_policy import instantiate_model


def run(config, rank, world_size):
    device = f"cuda:{rank}"
    writer = instantiate(config.exp.logger)
    writer.open(config)

    transforms = instantiate(config.transforms)

    data_files = writer.load_dataset(config.dataset)
    dataset = OfflineEpisodeDataset(transforms, files=data_files)
    encoder = instantiate_model(config.vis_encoder)
    encoder = encoder.to(device)

    total = len(dataset)
    i = rank

    dirpath = os.path.join(
        config.generator.store_artifact.dirpath, config.generator.store_artifact.name
    )
    os.makedirs(dirpath, exist_ok=True)

    pbar = tqdm(
        total=math.ceil(total / world_size), desc="Computing embeddings for episodes..."
    )
    while i < total:
        current_batch_size = 0
        episodes = []
        indices = []

        while current_batch_size < 2048 and i < total:
            if not os.path.exists(os.path.join(dirpath, f"data.{i}.pt")):
                episodes.append(dataset[i])
                indices.append(i)
                current_batch_size += episodes[-1]["rgb"].shape[0]

            i += world_size

        rgbs_t = torch.cat([episode["rgb"] for episode in episodes], dim=0)
        goals_t = torch.cat([episode["imagegoal"] for episode in episodes], dim=0)

        def apply_fn_in_batches(tensor, fn, max_batch_size):
            out = list(
                map(
                    lambda j: fn(tensor[j : j + max_batch_size]),
                    range(0, tensor.shape[0], max_batch_size),
                )
            )
            out = torch.cat(out)
            return out

        x = torch.cat([rgbs_t, goals_t], dim=0)
        x = x.to(device)
        x = einops.rearrange(x, "t h w c -> t c h w")
        x = apply_fn_in_batches(x, encoder.preprocess_transform, 3096)
        with encoder.maybe_autocast():
            x = encoder.backbone(x)
            x = x.last_hidden_state
        x = x.to(torch.bfloat16)
        x = x.cpu()

        rgbs_e, goals_e = torch.split(x, [rgbs_t.shape[0], goals_t.shape[0]])
        rgbs_e = torch.split(rgbs_e, [episode["rgb"].shape[0] for episode in episodes])
        goals_e = torch.split(
            goals_e, [episode["imagegoal"].shape[0] for episode in episodes]
        )

        for j in range(len(episodes)):
            episodes[j]["rgb"] = rgbs_e[j].clone()
            episodes[j]["imagegoal"] = goals_e[j].clone()
            del episodes[j]["depth"]

            torch.save(
                episodes[j],
                os.path.join(
                    dirpath,
                    f"data.{indices[j]}.pt",
                ),
            )
        pbar.update(len(episodes))
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument("cfg_path", type=str, help="Path to the configuration file")
    parser.add_argument("rank", type=int, help="Rank of this program")
    parser.add_argument("world_size", type=int, help="World size of this program")
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    run(config, args.rank, args.world_size)


if __name__ == "__main__":
    main()
