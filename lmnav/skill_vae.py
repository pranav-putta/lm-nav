import os
import editdistance

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import *

from random import randrange

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VanillaVAE(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 temperature: float,
                 kl_weight: float,
                 num_layers: int,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.temperature = temperature

        self.kld_weight = kl_weight

        self.embedding = nn.Embedding(self.num_tokens, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.dec_embedding = nn.Embedding(self.num_tokens, hidden_dim)
        self.dec_latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_head = nn.Linear(hidden_dim, vocab_size)

    @property
    def start_token(self):
        return self.vocab_size

    @property
    def pad_token(self):
        return self.vocab_size + 1

    @property
    def num_tokens(self):
        return self.vocab_size + 2

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        out = self.embedding(input)
        out, hx = self.encoder(out)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(hx[-1])
        log_var = self.fc_var(hx[-1])

        return [mu, log_var]

    def decode(self, z: Tensor, seq_len) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        hx = z.expand(self.num_layers, -1, -1)
        hx = self.dec_latent_proj(hx)
        inputs = torch.tensor([[self.start_token] * z.shape[0]]).T.cuda()

        out_seq = []
        for _ in range(seq_len):
            inputs = self.dec_embedding(inputs)
            out, hx = self.decoder(inputs, hx)
            logits = self.decoder_head(out) / self.temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1).long()
            out_seq.append(probs)
            inputs = next_token.detach().clone()

        return torch.stack(out_seq, dim=1)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, input.shape[1]), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0].view(-1, self.vocab_size)
        input = args[1].view(-1)
        mu = args[2]
        log_var = args[3]

        recons_loss = F.cross_entropy(recons, input, ignore_index=self.pad_token)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach().item(), 'KLD': -kld_loss.detach().item()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class ModeDataset(Dataset):
    
    def __init__(self):
        r = lambda length: [randrange(0, 4) for _ in range(length)]
        num_modes = 4
        self.modes = [r(l) for l in [randrange(10, 20) for _ in range(num_modes)]]
        self.modes = [[0] * 12, [0, 1] * 5, [1, 2] * 5 + [0, 1]] 
        print(self.modes)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return torch.tensor(self.modes[index % len(self.modes)])
        
class ActionDataset(Dataset):

    def __init__(self, data_path=None, bostoken=4, min_len=8, max_len=40):
        assert (data_path is not None)
        
        # set metadata using the config file
        self.min_len = min_len
        self.max_len = max_len
        self.bostoken = bostoken
        
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, actions = self.data[idx]
        actions = actions.tolist()
        # add a BOS token
        actions = [self.bostoken] + actions
        rand_len = min(randrange(self.min_len, self.max_len), len(actions)) 
        start = randrange(0, len(actions) - rand_len + 1)
        actions = actions[start:start + rand_len]
        return torch.tensor(actions, dtype=torch.long)

vocab_size = 4
embedding_dim = 8
hidden_dim = 64
latent_dim = 3
temp = 0.95
num_layers = 2

# dataset = ActionDataset("/srv/flash1/pputta7/projects/lm-nav/action_dataset.pt")
dataset = ModeDataset()
dataloader = DataLoader(dataset,
                        collate_fn=lambda t: pad_sequence(t, batch_first=True, padding_value=vocab_size + 1),
                        batch_size=16, num_workers=1)
model = VanillaVAE(vocab_size, embedding_dim, hidden_dim, latent_dim, temp, 0.05, num_layers)
model = model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Num params", sum([p.numel() for p in model.parameters()]))
for epoch in range(5):
    step = 0
    for actions in (pbar := tqdm(dataloader)):
        optim.zero_grad()
        out = model(actions.cuda())
        loss = model.loss_function(*out)

        loss['loss'].backward()
        optim.step()

        pbar.set_description(f"{loss}")
        step += 1
        if step % 25 == 0:
            torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, 'skill_vae2.pt')

    print(f"Epoch {epoch} eval:")
    for i in range(3):
        pred, inp, _, _ = model(dataset[i][None,].cuda())
        pred = torch.argmax(pred, dim=-1)
        pred = ' '.join(map(lambda x: str(x), pred.squeeze().tolist()))
        inp = ' '.join(map(lambda x: str(x), inp.squeeze().tolist()))
        print(f"Input: {inp} ; Pred: {pred} ; Distance: {editdistance.eval(pred, inp)}" )
    
