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
        act = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
        return act

