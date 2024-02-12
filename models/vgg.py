import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from metrics.metric import r2_score
from torchvision.models.vgg import vgg16_bn
from models.linear_models.fully_connect import FullyConnected


class VGG(nn.Module):

    def __init__(self, in_channels):
        super(VGG, self).__init__()

        # units = [[64, 2], [128, 2], [256, 4], [512, 4], [512, 4]]
        # layers = []
        #
        # for h, u in units:
        #     for _ in range(u):
        #         layers.append(nn.Sequential(nn.Conv2d(in_channels, h, 3, padding=1),
        #                                     nn.BatchNorm2d(h),
        #                                     nn.LeakyReLU()))
        #         in_channels = h
        #
        #     layers.append(nn.MaxPool2d(2))

        pre_trained_w = vgg16_bn(pretrained=True)
        self.feat_extractor = nn.Sequential(*(list(pre_trained_w.children())[:-2]))
        self.fc_branch = FullyConnected(487, 512)
        self.avgpool = nn.AvgPool2d(3)
        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(),
                                        nn.Linear(512, 6))

    def forward(self, x):
        x1 = self.feat_extractor(x[0])
        x1 = self.avgpool(x1)
        b, n, w, h = x1.shape
        x1 = x1.view(b, n * w * h)

        x2 = self.fc_branch(x[1])
        x_concat = torch.concat((x1, x2), dim=1)

        main = self.classifier(x_concat)
        aux = self.classifier(x_concat)

        return main, aux


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data', '../data/processed')
    x, y = next(iter(DataLoader(dataset, 32, True)))
    m = VGG(3)
    main, aux = m(x)

    loss = nn.MSELoss()
    loss_score = 0.5 * loss(main, y[0]) + 0.5 * loss(aux, y[1])

    scores = []

    # for _ in range(6):
    #     scores.append(r2_score(y[:, _].detach().numpy(), r[:, _].detach().numpy()))

    # print(np.mean(scores))
    print(main.shape, aux.shape)
    print(loss_score)
