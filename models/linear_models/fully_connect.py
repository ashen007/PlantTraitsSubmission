import torch
from torch import nn


class FullyConnected(nn.Module):

    def __init__(self, in_units, out_units):
        super(FullyConnected, self).__init__()

        layers = []

        for _ in range(3):
            layers.append(nn.Sequential(nn.Linear(in_units, out_units),
                                        nn.LeakyReLU(),
                                        nn.BatchNorm1d(out_units),
                                        nn.Dropout(0.2))
                          )

            in_units = out_units

        self.feat_extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feat_extractor(x)
