import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class CustomEffnet(nn.Module):
    def __init__(self):
        super(CustomEffnet, self).__init__()

        self.model = timm.create_model('timm/efficientvit_l2.r384_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data/2024')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    # print(x.shape, y.shape)

    m = CustomEffnet()
    main = m(x)

    loss = nn.MSELoss()

    loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(loss_score)
