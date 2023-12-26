from torch import nn
from lmnav.common.utils import is_url, convert_weights_to_fp16


class LinearHead(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=None, in_dim=1024, p_dropout=0.2, **kwargs):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.dropout = nn.Dropout(p_dropout)
        
        # add linear layers
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.summary = nn.Sequential(*layers)

        # convert_weights_to_fp16(self.summary)

    def forward(self, x):
        output = self.dropout(x)
        output = self.summary(x)
        return output
