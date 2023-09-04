import torch
import einops
import random

idx2class = {
    0: 'stop',
    1: 'forward',
    2: 'left',
    3: 'right'
}

def apply_transforms_inputs(vis_processor, rgbs, goals, actions):
    """
    rgbs: torch.Tensor[B T C H W] -> [B C T H W]
    goals: torch.Tensor[B 1 C H W] -> [B C 1 H W]
    actions: torch.Tensor[B T] -> list[list[str]]
    """
    B, T, _, _, _ = rgbs.shape
    imgs = torch.cat([goals, rgbs], dim=1)
    imgs = einops.rearrange(imgs, 'b t c h w -> c (b t) h w')
    imgs = imgs.float()

    # apply transform
    imgs = vis_processor.transform(imgs)
    imgs = einops.rearrange(imgs, 'c (b t) h w -> b c t h w', b=B)

    # separate goal and rgb
    goals, rgbs = imgs[:, :, 0:1], imgs[:, :, 1:]

    # process actions into tokens
    actions = [[idx2class[act] for act in acts_t] for acts_t in actions]

    return rgbs, goals, actions


def extract_inputs_from_dataset(dataset):
    goals = [torch.from_numpy(episode[0]['observation']['imagegoal']) for episode in dataset]
    goals = [einops.rearrange(goal, 'h w c -> 1 c h w') for goal in goals]

    rgbs = [[torch.from_numpy(state['observation']['rgb']) for state in episode] for episode in dataset]
    rgbs = [torch.stack(rgb_e) for rgb_e in rgbs]
    rgbs = [einops.rearrange(rgb_t, 't h w c -> c t h w') for rgb_t in rgbs]

    actions = [torch.tensor([state['action'] for state in episode]) for episode in dataset]
    
    return rgbs, goals, actions
    

def sample_subsequences(B, T, rgbs, goals, actions):
    """
    rgbs: list[torch.Tensor[T, C, H, W]] -> torch.Tensor[B, T, C, H, W]
    goals: list[torch.Tensor[1, C, H, W]] -> torch.Tensor[B, 1, C, H, W ]
    actions: list[torch.Tensor[T]] -> torch.Tensor[B, T]
    """

    episode_lens = [ep_rgb.shape[0] for ep_rgb in rgbs]
    n_episodes = len(rgbs)

    rgbs_t, goals_t, actions_t = [], [], []
    
    sliding_windows = [[slice(start, start + T, 1) for start in range(length - T + 1)] for length in episode_lens] 
    
    while len(rgbs_t) < B:
        # TODO; maybe don't construct tensors here
        i = random.choices(range(n_episodes), weights=episode_lens, k=1)[0]
        length = episode_lens[i]
        if length < T:
            continue
        slices = random.sample(sliding_windows[i], 1)

        rgbs_t += [rgbs[i][s].clone() for s in slices]
        goals_t += [goals[i].clone() for _ in slices]
        actions_t += [actions[i][s].clone() for s in slices]
    
    rgbs_t = torch.stack(rgbs_t, dim=0)      
    goals_t = torch.stack(goals_t, dim=0)      
    actions_t = torch.stack(actions_t, dim=0)

    return rgbs_t, goals_t, actions_t

