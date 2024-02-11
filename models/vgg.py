import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from metrics.metric import r2_score
from torchvision.models.vgg import vgg16


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
        #                                     nn.ReLU()))
        #         in_channels = h
        #
        #     layers.append(nn.MaxPool2d(2))

        self.feat_extractor = vgg16(pretrained=True)
        self.feat_extractor.avgpool = nn.AvgPool2d(3)
        self.feat_extractor.classifier = nn.Sequential(nn.Flatten(),
                                                       nn.Linear(512, 6))

    def forward(self, x):
        x = self.feat_extractor(x)

        return x


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data', '../data/processed')
    x, y = next(iter(DataLoader(dataset, 16, True)))
    m = VGG(3)
    r = m(x)

    loss = nn.MSELoss()
    loss_score = loss(r, y)

    scores = []

    for _ in range(6):
        scores.append(r2_score(y[:, _].detach().numpy(), r[:, _].detach().numpy()))

    print(np.mean(scores))
    print(r.shape)
    print(loss_score)
