import timm
import pandas as pd
from torch import nn
from dataloader.transformers import INPAINT_TRANSFORMER
from dataloader.imageInpaintData import ImageDataset
from torch.utils.data import DataLoader


# class Discriminator(nn.Module):
#
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.backbone = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1),
#                                       nn.LeakyReLU(),
#                                       nn.Conv2d(64, 128, 3, 2, 1),
#                                       nn.InstanceNorm2d(128),
#                                       nn.LeakyReLU(),
#                                       nn.Conv2d(128, 256, 3, 2, 1),
#                                       nn.InstanceNorm2d(256),
#                                       nn.LeakyReLU(),
#                                       nn.Conv2d(256, 512, 3, 2, 1),
#                                       nn.InstanceNorm2d(512),
#                                       nn.LeakyReLU(),
#                                       nn.Conv2d(512, 1, 3, 1, 1)
#                                       )
#
#     def forward(self, x):
#         x = self.backbone(x)
#
#         return x


class FeatureLearnModule(nn.Module):

    def __init__(self):
        super(FeatureLearnModule, self).__init__()

        backbone = timm.create_model('tresnet_m.miil_in21k_ft_in1k',
                                     pretrained=True,
                                     num_classes=0)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))
        self.decoder = nn.Sequential(nn.Conv2d(2048, 1024, 3, 2, 1),
                                     nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                                     nn.BatchNorm2d(512),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 3, 3, 1, 1),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = ImageDataset(df, INPAINT_TRANSFORMER)
    x, y, aux = next(iter(DataLoader(dataset, 4, True)))

    m = FeatureLearnModule()
    # d = Discriminator()
    main = m(x)
    # dis = d(aux)

    print(main.shape)
    # print(dis.shape)
