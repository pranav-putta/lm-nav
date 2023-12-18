import torch

class MaximumLikelihoodSampler:
    def __init__(self):
        pass

    def __call__(self, logits):
        return torch.argmax(logits, dim=-1).item()

class NucleusSampler:
    def __init__(self, p=1.0, temp=1.0, seed=None):
        self.p = p
        self.temp = temp
        self.generator = torch.Generator(device='cuda')
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, logits):
        probs = torch.softmax(logits / self.temp, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = cumulative_probs > self.p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0
        sorted_probs[indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        selected_idx = torch.multinomial(sorted_probs, num_samples=1, generator=self.generator).item()
        act = sorted_indices[selected_idx].item()
        return act

