import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from models.backbone.danet import DANetHead
from models.linear_models.recurrent import RecurrentTraitNet
from models.backbone.residual_attention_network import \
    ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from models.backbone.lstm import TraitLSTM


class ExpNet(nn.Module):

    def __init__(self, in_channels):
        super(ExpNet, self).__init__()

        # network branches
        self.net_head = nn.Sequential(nn.Conv2d(in_channels, 16, 7, 2, 3),
                                      nn.BatchNorm2d(16),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(16, 16, 5, 2, 2),
                                      nn.BatchNorm2d(16),
                                      nn.LeakyReLU(),
                                      )
        self.rnn_head = nn.Conv2d(16, 4, 3, 1, 1)

        self.rat_branch = ResidualAttentionModel()
        self.rnn_branch = TraitLSTM(32, 256, 2, 256)
        self.fc_branch = RecurrentTraitNet(489, 256, 512, 2)

        self.regress = nn.Sequential(nn.Linear(3072, 1024),
                                     nn.BatchNorm1d(1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, 6))

    def forward(self, x):
        x_0 = self.net_head(x[0])
        x_0_rat = self.rat_branch(x_0)

        x_0_rnn = self.rnn_branch(self.rnn_head(x_0).view(-1, 32, 32)).view(32, -1)
        x_1 = self.fc_branch(x[1])

        x_0_rat = torch.concat((x_0_rat, x_1), dim=1)
        x_0_rnn = torch.concat((x_0_rnn, x_1), dim=1)

        x_concat = torch.concat((x_0_rat, x_0_rnn), dim=1)

        main = self.regress(x_concat)
        aux = self.regress(x_concat)

        return main, aux


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data', '../data/processed')
    x, y = next(iter(DataLoader(dataset, 32, True)))
    m = ExpNet(3)
    main, aux = m(x)

    loss = nn.MSELoss()
    loss_score = 0.5 * loss(main, y[0]) + 0.5 * loss(aux, y[1])

    print(main.shape, aux.shape)
    print(loss_score)
