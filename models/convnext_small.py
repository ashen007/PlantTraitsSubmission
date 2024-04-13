import timm
import pandas as pd
from torch import nn
from dataloader.transformers import TRANSFORMER
from dataloader.PlantTriatData import PlantDataset
from torch.utils.data import DataLoader


class ConvNext(nn.Module):
    def __init__(self):
        super(ConvNext, self).__init__()

        self.model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = PlantDataset(df, TRANSFORMER)
    x, y = next(iter(DataLoader(dataset, 16, True)))

    m = ConvNext()
    main = m(x)

    print(main.shape)
