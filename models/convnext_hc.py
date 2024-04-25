import timm
import pandas as pd
from torch import nn
from dataloader.transformers import TRANSFORMER
from dataloader.PlantTriatData import PlantDataset
from torch.utils.data import DataLoader


class ConvNext(nn.Module):
    def __init__(self):
        super(ConvNext, self).__init__()

        self.model = timm.create_model('convnext_small.in12k_ft_in1k_384',
                                       pretrained=True,
                                       num_classes=30)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    df = pd.read_csv('../data/train_23_hc.csv')
    dataset = PlantDataset(df, TRANSFORMER)
    x, y = next(iter(DataLoader(dataset, 4, True)))

    m = ConvNext()
    main = m(x)

    print(main.shape)
