import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class ResNest(nn.Module):
    def __init__(self):
        super(ResNest, self).__init__()

        self.model = timm.create_model('resnest50d_4s2x40d.in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../data/2024/')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    # print(x.shape, y.shape)

    m = ResNest()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
