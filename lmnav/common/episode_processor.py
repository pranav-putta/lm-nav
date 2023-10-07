import torch
import einops
import random
import math

import torch.nn.functional as F


def apply_transforms_actions(actions):
    return actions


def apply_transforms_images(vis_processor, rgbs, goals):
    B, T, _, _, _ = rgbs.shape
    imgs = torch.cat([goals, rgbs], dim=1)
    imgs = einops.rearrange(imgs, "b t c h w -> c (b t) h w")
    imgs = imgs.float()

    # apply transform
    imgs = vis_processor.transform(imgs)
    imgs = einops.rearrange(imgs, "c (b t) h w -> b c t h w", b=B)

    # separate goal and rgb
    goals, rgbs = imgs[:, :, 0:1], imgs[:, :, 1:]
    return rgbs, goals


def apply_transforms_inputs(vis_processor, rgbs, goals, actions):
    """
    rgbs: torch.Tensor[B T C H W] -> [B C T H W]
    goals: torch.Tensor[B 1 C H W] -> [B C 1 H W]
    actions: torch.Tensor[B T] ->
    """
    rgbs, goals = apply_transforms_images(vis_processor, rgbs, goals)
    actions = apply_transforms_actions(actions)

    return rgbs, goals, actions


def extract_inputs_from_dataset(dataset):
    goals = [episode[0]["observation"]["imagegoal"] for episode in dataset]
    goals = [einops.rearrange(goal, "h w c -> 1 c h w") for goal in goals]

    rgbs = [[state["observation"]["rgb"] for state in episode] for episode in dataset]
    rgbs = [torch.stack(rgb_e) for rgb_e in rgbs]
    rgbs = [einops.rearrange(rgb_t, "t h w c -> t c h w") for rgb_t in rgbs]

    actions = [
        torch.tensor([state["action"] for state in episode]) for episode in dataset
    ]

    return rgbs, goals, actions


def construct_subsequences(B, T, rgbs, goals, actions):
    """
    rgbs: list[torch.Tensor[T, C, H, W]] -> torch.Tensor[B, T, C, H, W]
    goals: list[torch.Tensor[1, C, H, W]] -> torch.Tensor[B, 1, C, H, W ]
    actions: list[torch.Tensor[T]] -> torch.Tensor[B, T]
    """

    subsequences = []

    n_episodes = len(rgbs)
    episode_lens = [int(ep_rgb.shape[0]) for ep_rgb in rgbs]

    # set T to smaller length if there is a shortest episode
    sliding_windows = [
        [
            slice(start, start + min(T, length), 1)
            for start in range(length - min(T, length) + 1)
        ]
        for length in episode_lens
    ]

    # compute the number of samples for each episode
    window_lengths = [len(sw) for sw in sliding_windows]
    episode_weights = torch.tensor(window_lengths) / sum(window_lengths)
    samples_per_episode = [
        min(window_lengths[i], math.ceil(B * episode_weights[i]))
        for i in range(n_episodes)
    ]

    # B can be at most the window lengths so artificially cut it off
    B = min(B, sum(window_lengths))

    # make sure constraint of sum(samples) = B is still satisfied
    i = 0
    delta = 1 if sum(samples_per_episode) < B else -1
    while sum(samples_per_episode) != B:
        samples_per_episode[i] = max(
            min(window_lengths[i], samples_per_episode[i] + delta), 1
        )
        i = (i + 1) % len(samples_per_episode)

    # now, num_samples_per_episode are num_windows <= N <= B

    for i in range(n_episodes):
        num_samples = samples_per_episode[i]

        slices = random.sample(sliding_windows[i][:-1], num_samples - 1)
        # make sure the last subsequence with STOP is always included
        slices.append(sliding_windows[i][-1])
        subsequences += [
            (rgbs[i][s].clone(), goals[i].clone(), actions[i][s].clone())
            for s in slices
        ]

    random.shuffle(subsequences)
    rgbs, goals, actions = zip(*subsequences)

    return rgbs, goals, actions
