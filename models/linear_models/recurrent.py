import torch
from torch import nn


class RecurrentTraitNet(nn.Module):

    def __init__(self, in_units, hidden_units, out_units, n_layers=2):
        super(RecurrentTraitNet, self).__init__()

        self.hidden_size = hidden_units
        self.layer_size = n_layers
        self.lstm = nn.LSTM(in_units, hidden_units, n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_units, out_units)

    def forward(self, x):
        # Hidden state:
        hidden_state = torch.zeros(self.layer_size * 2, x.size(0), self.hidden_size, device='cuda')
        # Cell state:
        cell_state = torch.zeros(self.layer_size * 2, x.size(0), self.hidden_size, device='cuda')

        lstm_out, _ = self.lstm(x.view(len(x), 1, -1), (hidden_state, cell_state))
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)

        return predictions
