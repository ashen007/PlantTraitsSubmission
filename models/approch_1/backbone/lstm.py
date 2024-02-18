import torch
from torch import nn


class TraitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True):
        super(TraitLSTM, self).__init__()

        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional

        # Step1: the LSTM model
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=bidirectional)

        # Step2: the FNN
        self.layer = nn.Linear(hidden_size * 2, output_size)

    def forward(self, images):
        # Hidden state:
        hidden_state = torch.zeros(self.layer_size * 2, images.size(0), self.hidden_size, device='cuda')
        # Cell state:
        cell_state = torch.zeros(self.layer_size * 2, images.size(0), self.hidden_size, device='cuda')

        # LSTM:
        output, (last_hidden_state, last_cell_state) = self.lstm(images, (hidden_state, cell_state))

        # Reshape
        output = output[:, -1, :]

        # FNN:
        output = self.layer(output)

        return output
