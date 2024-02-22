import timm
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class CustomEffnetLarge(nn.Module):
    def __init__(self):
        super(CustomEffnetLarge, self).__init__()

        self.model = timm.create_model('efficientvit_b1.r288_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../../data', '../../data/processed')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    print(x.shape, y.shape)

    m = CustomEffnetLarge()
    main = m(x)

    loss = nn.SmoothL1Loss()
    loss_score = loss(main, y)

    print(main)
    print(main.shape)
    print(loss_score)
