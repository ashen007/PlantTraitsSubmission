import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from loss import R2Loss, AltR2Loss


class CustomEffnetMM(nn.Module):
    def __init__(self):
        super(CustomEffnetMM, self).__init__()

        self.model = timm.create_model('efficientvit_b2.r288_in1k',
                                       pretrained=True,
                                       num_classes=16)

        self.reg = nn.Sequential(nn.Linear(163, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 #                          nn.Linear(1024, 1024),
                                 #                          nn.LayerNorm(1024),
                                 #                          nn.ReLU(),
                                 #                          nn.Dropout(0.4),
                                 nn.Linear(128, 16))

        self.out_layer = nn.Sequential(nn.Linear(32, 16),
                                       nn.BatchNorm1d(16),
                                       nn.ReLU(),
                                       nn.Linear(16, 6))

    def forward(self, x):
        x_ = self.model(x[0])
        x_reg = self.reg(x[1])

        x_con = torch.concat((x_, x_reg), dim=1)

        out = self.out_layer(x_con)

        return out


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    print(x[0].shape, y.shape)

    m = CustomEffnetMM()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(main)
    # print(loss_score)
