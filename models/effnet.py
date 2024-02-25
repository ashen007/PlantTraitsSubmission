import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from loss import R2Loss, AltR2Loss


class CustomEffnet(nn.Module):
    def __init__(self):
        super(CustomEffnet, self).__init__()

        self.model = timm.create_model('efficientvit_b1.r288_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    print(x.shape, y.shape)

    m = CustomEffnet()
    main = m(x)

    loss = nn.MSELoss()

    loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(loss_score)
