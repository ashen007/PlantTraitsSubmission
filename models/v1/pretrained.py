import timm
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class ConvNext(nn.Module):
    def __init__(self):
        super(ConvNext, self).__init__()

        self.model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)
        # x_ = self.fc(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../data')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    # print(x.shape, y.shape)

    m = ConvNext()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
