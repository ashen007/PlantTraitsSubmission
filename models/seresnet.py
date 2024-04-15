import timm
import pandas as pd
from torch import nn
from dataloader.transformers import TRANSFORMER
from dataloader.PlantTriatData import PlantDataset
from torch.utils.data import DataLoader


class SeResNet(nn.Module):
    def __init__(self):
        super(SeResNet, self).__init__()

        self.model = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = PlantDataset(df, TRANSFORMER)
    x, y = next(iter(DataLoader(dataset, 32, True)))

    m = SeResNet()
    main = m(x)

    print(main.shape)
