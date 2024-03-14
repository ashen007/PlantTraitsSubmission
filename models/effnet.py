import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from loss import R2Loss, AltR2Loss


class CustomEffnet(nn.Module):
    def __init__(self):
        super(CustomEffnet, self).__init__()

        self.model = timm.create_model('efficientvit_b2.r288_in1k',
                                       pretrained=True,
                                       num_classes=0)
        self.reg = nn.Sequential(nn.Linear(384, 1024),
                                 nn.LayerNorm(1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 1024),
                                 nn.LayerNorm(1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(1024, 6))

    def forward(self, x):
        x_ = self.model(x[0])
        x_ = self.reg(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    # print(x.shape, y.shape)

    m = CustomEffnet()
    main = m(x)

    loss = nn.MSELoss()

    loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(loss_score)
