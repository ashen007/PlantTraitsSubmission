import timm
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class CustomConvNext(nn.Module):
    def __init__(self):
        super(CustomConvNext, self).__init__()

        self.model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x[0])

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    # print(x.shape, y.shape)

    m = CustomConvNext()
    main = m(x)

    loss = nn.SmoothL1Loss()
    loss_score = loss(main, y)

    print(main)
    print(main.shape)
    print(loss_score)

