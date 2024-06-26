import timm
import torch
import pandas as pd
from torch import nn
from dataloader.PlantTriatData import PlantDataset
from dataloader.transformers import TRANSFORMER
from torch.utils.data import DataLoader


class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()

        self.model = timm.create_model('efficientvit_l2.r288_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = PlantDataset(df, TRANSFORMER)
    x, y = next(iter(DataLoader(dataset, 16, True)))

    m = EffNet()
    main = m(x)

    print(main.shape)
