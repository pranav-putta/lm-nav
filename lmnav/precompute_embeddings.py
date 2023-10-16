import argparse
import torch


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

        while current_batch_size < 3096 and i < total:
            episodes.append(dataset[i])
            current_batch_size += episodes[-1]["rgb"].shape[0]
            i += world_size

        goals_e, rgbs_e = encoder.embed_obs(episodes)

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
    args = parser.parse_args()

    config = get_config(args.cfg_path)
    run(config)


if __name__ == "__main__":
    main()
