from torch import nn
from lmnav.common.utils import is_url, convert_weights_to_fp16


class LinearHead(nn.Module):
    def __init__(self, in_dim=None, p_dropout=0.2, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.summary = nn.Linear(in_dim, 1)

        # convert_weights_to_fp16(self.summary)

    def forward(self, x):
        output = self.dropout(x)
        output = self.summary(x)
        return output
