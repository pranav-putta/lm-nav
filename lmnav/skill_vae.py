import random
from random import randrange
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

import editdistance
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from einops import repeat, rearrange


class Encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, hidden = self.lstm(x)
        return hidden


class Decoder(nn.Module):
    def __init__(
            self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, hidden = self.lstm(x)
        prediction = self.fc(output)
        return prediction, hidden


class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
            self, vocab_size, embedding_size, hidden_size, latent_size,
            num_layers, device=torch.device("cuda"), kl_weight=0.00025
    ):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size of LSTM AE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMVAE, self).__init__()
        self.device = device

        # dimensions
        self.kl_weight = kl_weight
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # lstm ae
        self.embedding = nn.Embedding(vocab_size + 1, self.embedding_size)
        self.lstm_enc = Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            input_size=latent_size,
            output_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        self.fc21 = nn.Linear(self.hidden_size * self.num_layers * 2, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size * self.num_layers * 2, self.latent_size)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def forward(self, inp):
        batch_size, seq_len = inp.shape

        x = self.embedding(inp)

        # encode input space to hidden space
        enc_hidden = self.lstm_enc(x)
        hidden = rearrange(enc_hidden, 'l b h -> b (l h)')

        # extract latent variable z(hidden space to latent space)
        mean = self.fc21(hidden)
        logvar = self.fc22(hidden)
        z = self.reparametize(mean, logvar)  # batch_size x latent_size

        original_z = z
        
        # decode latent space to input space
        z = repeat(z, 'b h -> b l h', l=seq_len)
        
        reconstruct_output, hidden = self.lstm_dec(z)
        
        x_hat = reconstruct_output

        # calculate vae loss
        losses = self.loss_function(x_hat, inp, mean, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, x_hat, (recon_loss, kld_loss, original_z)

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = self.kl_weight  # Account for the minibatch samples from the dataset
        recons_loss = F.cross_entropy(recons.view(-1, self.vocab_size), input.view(-1), ignore_index=self.vocab_size)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }


class ModeDataset(Dataset):

    def __init__(self):
        r = lambda length: [randrange(0, 4) for _ in range(length)]
        num_modes = 4
        random.seed(0)
        self.modes = [r(l) for l in [100] * 4]
        # self.modes = [[0, 1, 2] * 10, [0, 1, 0] * 10]
        print(self.modes)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return torch.tensor(self.modes[index % len(self.modes)])

    def val2str(self, val):
        return ''.join(map(str, val.squeeze().cpu().tolist()))


def plot_and_cluster_features(tensors, k, title):
    # Performing kmeans clustering
    centroids, distortion = kmeans(tensors, k)
    idx, _ = vq(tensors, centroids)

    # Plot the clustered data
    for i in range(k):
        plt.scatter(tensors[idx == i, 0], tensors[idx == i, 1], label=f'Cluster {i + 1}')

    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color='black', label='Centroids')
    plt.title("t-SNE of latents")

    plt.legend()
    plt.savefig(f'{title}.png')
    plt.clf()


def plot_and_reduce_dimensions(data, title):
    # Extract the tensors and labels from the data
    tensors = torch.cat([t for t, _ in data], dim=0).cpu().numpy()
    labels = [l for _, l in data]

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=0)
    tensors_2d = tsne.fit_transform(tensors)

    # Plot the reduced data
    plot_and_cluster_features(tensors_2d, 8, title)

class ActionDataset(Dataset):

    def __init__(self, data_path=None, min_len=8, max_len=40):
        assert (data_path is not None)

        # set metadata using the config file
        self.min_len = min_len
        self.max_len = max_len

        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, actions = self.data[idx]
        actions = actions.tolist()
        # add a BOS token
        rand_len = min(randrange(self.min_len, self.max_len), len(actions))
        start = randrange(0, len(actions) - rand_len + 1)
        actions = actions[start:start + rand_len]
        return torch.tensor(actions, dtype=torch.long) 

    def val2str(self, val):
        return ' '.join(map(str, val.squeeze().cpu().tolist()))

def train():
    vocab_size = 4
    embedding_dim = 64
    hidden_dim = 64
    latent_dim = 512
    temp = 1.0
    num_layers = 4
    lr = 1e-4
    kl_weight = 0.05
    device = torch.device('cuda')

    dataset = ActionDataset("/srv/flash1/pputta7/projects/lm-nav/actdataset.pt")
    # dataset = ModeDataset()
    dataloader = DataLoader(dataset,
                            collate_fn=lambda t: pad_sequence(t, batch_first=True, padding_value=vocab_size),
                            batch_size=16)
    model = LSTMVAE(vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, device, kl_weight=kl_weight)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print("Num params", sum([p.numel() for p in model.parameters()]))
    for epoch in range(1000):
        step = 0
        avg_loss, avg_recon_loss, avg_kl_loss = 0, 0, 0
        for actions in (pbar := tqdm(dataloader)):
            optim.zero_grad()
            loss, out, extra = model(actions.to(device))

            loss.backward()
            optim.step()
            
            avg_loss += loss
            avg_recon_loss += extra[0]
            avg_kl_loss += extra[1]
            
            
            step += 1
            pbar.set_description(f"Loss: {avg_loss / step:.3f} ; Reconstruction Loss: {avg_recon_loss / step:.3f} ; KDL: {avg_kl_loss / step:.3f}")
            if step % 25 == 0:
                torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, f'skill_vae_epoch={epoch}.pt')

        print(f"Epoch {epoch} eval:")
        latents = []
        avg_edit_dist = 0
        with torch.no_grad():
            for i in range(1000):
                inp = dataset[i][None,].to(device)
                _, pred, (_, _, z) = model(inp.to(device))

                pred = torch.argmax(pred, dim=-1)
                label = dataset.val2str(pred)
                inp = dataset.val2str(inp)
                
                dist = editdistance.eval(label, inp)
                avg_edit_dist += dist
                if i % 250 == 0:
                    print(f"Input: {inp}, Pred: {label}, Distance: {dist}")
                latents.append((z, label))
            print("Average edit distance", avg_edit_dist / len(latents))
            # plot_and_reduce_dimensions(latents, f'epoch={epoch}')

if __name__ == "__main__":
    train()
