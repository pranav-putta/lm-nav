import argparse
import torch
import einops

from lmnav.config.default import get_config

from hydra.utils import instantiate

from lmnav.common.writer import *

from lmnav.dataset.datasets import OfflineEpisodeDataset
from lmnav.models.base_policy import instantiate_model


def run(config, rank, world_size):
    writer = instantiate(config.exp.logger)
    writer.open(config)

    transforms = instantiate(config.transforms)

    data_files = writer.load_dataset(config.dataset)
    dataset = OfflineEpisodeDataset(transforms, files=data_files)
    encoder = instantiate_model(config.vis_encoder)
    encoder = encoder.to("cuda")

    total = len(dataset)
    i = rank

    dirpath = os.path.join(config.generator.store_artifact.dirpath, "00774_clip")
    os.makedirs(dirpath, exist_ok=True)

    pbar = tqdm(total=total, desc="Computing embeddings for episodes...")
    while i < total:
        current_batch_size = 0
        episodes = []

        while current_batch_size < 2048 and i < total:
            if not os.path.exists(os.path.join(dirpath, f"data.{i}.pth")):
                episodes.append(dataset[i])
                current_batch_size += episodes[-1]["rgb"].shape[0]

            i += world_size

        rgbs_t = torch.cat([episode["rgb"] for episode in episodes], dim=0)
        goals_t = torch.cat([episode["imagegoal"] for episode in episodes], dim=0)

        def apply_fn_in_batches(tensor, fn, max_batch_size):
            out = list(
                map(
                    lambda i: fn(tensor[i : i + max_batch_size]),
                    range(0, tensor.shape[0], max_batch_size),
                )
            )
            out = torch.cat(out)
            return out

        x = torch.cat([rgbs_t, goals_t], dim=0)
        x = x.to("cuda")
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

        for i in range(len(episodes)):
            episodes[i]["rgb"] = rgbs_e[i]
            episodes[i]["imagegoal"] = goals_e[i]
            del episodes[i]["depth"]

            torch.save(
                episodes[i],
                os.path.join(
                    dirpath,
                    f"data.{i}.pt",
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
