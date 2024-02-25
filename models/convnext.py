import timm
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class CustomConvNext(nn.Module):
    def __init__(self):
        super(CustomConvNext, self).__init__()

        self.model = timm.create_model('timm/convnext_small.fb_in22k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    print(x.shape, y.shape)

    m = CustomConvNext()
    main = m(x)

    loss = nn.MSELoss()
    loss_score = loss(main, y)

    print(main)
    print(main.shape)
    print(loss_score)

