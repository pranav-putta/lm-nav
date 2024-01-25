from typing import Optional
from habitat_baselines.common.tensor_dict import TensorDict
import torch
from collections import namedtuple

RolloutCache = namedtuple("RolloutCache", ["past_lengths", "past_kv_cache"]) 


class RolloutStorage:
    def __init__(self, num_envs, max_steps, tokens_per_img, hidden_size, device):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device

        self.rgbs = torch.zeros(
            num_envs,
            max_steps + 1,
            tokens_per_img,
            1024,
            dtype=torch.bfloat16,
            device=device,
        )
        self.goals = torch.zeros(
            num_envs,
            max_steps + 1,
            tokens_per_img,
            1024,
            dtype=torch.bfloat16,
            device=device,
        )
        self.prev_actions = torch.zeros(
            num_envs, max_steps + 1, dtype=torch.long, device=device
        )
        self.dones = torch.zeros(
            num_envs, max_steps + 1, dtype=torch.bool, device=device
        )
        self.successes = torch.zeros(
            num_envs, max_steps, dtype=torch.bool, device=device
        )
        self.rewards = torch.zeros(
            num_envs, max_steps + 1, dtype=torch.float, device=device
        )
        self.dtgs = torch.zeros(
            num_envs, max_steps + 1, dtype=torch.float, device=device
        )
        self.hidden_states = torch.zeros(
            num_envs, max_steps + 1, hidden_size, dtype=torch.bfloat16, device=device
        )
        self.logprobs = torch.zeros(
            num_envs, max_steps + 1, dtype=torch.float, device=device
        )

        self.current_step_idx = -1

        self.last_cache: Optional[RolloutCache] = None
        self.current_cache: Optional[RolloutCache] = None

    def insert(
        self,
        next_rgbs,
        next_goals,
        dones=None,
        actions=None,
        rewards=None,
        successes=None,
        dtgs=None,
        hx=None,
        logprobs=None,
    ):
        """insert observations, dones, and actions into rollout storage tensors"""
        self.rgbs[:, self.current_step_idx + 1] = next_rgbs
        self.goals[:, self.current_step_idx + 1] = next_goals

        if dones is not None:
            self.dones[:, self.current_step_idx] = dones
        if successes is not None:
            self.successes[:, self.current_step_idx] = successes
        if rewards is not None:
            self.rewards[:, self.current_step_idx] = rewards
        if actions is not None:
            self.prev_actions[:, self.current_step_idx + 1] = actions
        if dtgs is not None:
            self.dtgs[:, self.current_step_idx] = dtgs
        if hx is not None:
            self.hidden_states[:, self.current_step_idx] = hx
        if logprobs is not None:
            self.logprobs[:, self.current_step_idx] = logprobs

        self.current_step_idx += 1

    def reset(self):
        self.rgbs[:, 0] = self.rgbs[:, -1]
        self.goals[:, 0] = self.goals[:, -1]
        self.prev_actions[:, 0] = self.prev_actions[:, -1]
        
        self.current_step_idx = 0
        self.last_cache = RolloutCache(self.current_cache.past_lengths.clone().detach(), self.current_cache.past_kv_cache)        
        self.last_cache.past_lengths.masked_fill_(self.dones[:, self.max_steps - 1], 0)
        self.current_cache = None

    def generate_sample_idxs(self):
        """generate sample indices from the rollout storage"""
        sample_idxs = []
        for b in range(self.num_envs):
            ends = torch.where(self.dones[b])[0].tolist()
            # always make sure that ends includes the beginning/end of step list
            ends = [-1] + ends
            if ends[-1] != self.max_steps - 1:
                ends = ends + [self.max_steps - 1]

            slices = [(ends[i] + 1, ends[i + 1] + 1) for i in range(len(ends) - 1)]
            for i, j in slices:
                sample_idxs.append((b, i, j))
            
        return sample_idxs

        
    def generate_samples(self):
        """generate samples from the rollout storage"""
        samples = []

        for b in range(self.num_envs):
            ends = torch.where(self.dones[b])[0].tolist()
            # always make sure that ends includes the beginning/end of step list
            ends = [-1] + ends
            if ends[-1] != self.max_steps - 1:
                ends = ends + [self.max_steps - 1]

            slices = [(ends[i] + 1, ends[i + 1] + 1) for i in range(len(ends) - 1)]
            for i, j in slices:
                if (self.last_cache is not None) and i == 0:
                    sample_cache = self.last_cache.past_kv_cache[:, :, b, :, :self.last_cache.past_lengths[b]]
                else:
                    sample_cache = torch.zeros(32, 2, 32, 0, 128, device=self.device, dtype=torch.bfloat16)

                data = TensorDict(
                    rgb=self.rgbs[b, i:j],
                    goal=self.goals[b, i:i+1],
                    prev_action=self.prev_actions[b, i:j + 1],
                    reward=self.rewards[b, i:j],
                    done=self.dones[b, i:j],
                    success=self.successes[b, i:j],
                    dtg=self.dtgs[b, i:j],
                    hx=self.hidden_states[b, i:j],
                    logprobs=self.logprobs[b, i:j],
                )
                samples.append((data, sample_cache))

        return zip(*samples)

    def pad_samples(self, samples):
        keys = samples[0].keys()
        padded_samples = TensorDict()
        for k in keys:
            padded_samples[k] = torch.nn.utils.rnn.pad_sequence(
                [s[k] for s in samples], batch_first=True
            )
        return padded_samples

    def set_current_cache(self, past_lengths, past_kv_cache):
        past_kv_cache = torch.stack([torch.stack([past_kv_cache[i][0], past_kv_cache[i][1]]) for i in range(len(past_kv_cache))])
        past_kv_cache = past_kv_cache.to('cpu')
        self.current_cache = RolloutCache(past_lengths, past_kv_cache)


    def to_cpu(self):
        map(
            lambda t: t.cpu(),
            (self.rgbs, self.goals, self.dones, self.rewards, self.actions),
        )
