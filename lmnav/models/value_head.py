from torch import nn

class ValueHead(nn.Module):

    def __init__(self, in_dim, p_dropout):
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.summary = nn.Linear(in_dim, 1)

    def forward(self, x):
        output = self.dropout(x)
        output = self.summary(x)
        return output
