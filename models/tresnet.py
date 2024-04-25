import timm
import pandas as pd
import torch
from torch.nn import functional as F
from torch import nn
from dataloader.transformers import TRANSFORMER
from dataloader.PlantTriatData import PlantDataset
from torch.utils.data import DataLoader
from models.self_supervision_tresnet import FeatureLearnModule


class TResNet(nn.Module):
    def __init__(self):
        super(TResNet, self).__init__()

        feat_extractor = FeatureLearnModule()
        weights = torch.load('../learn/ckpts/ssl/best_ckpt_ssl_224.pth')
        feat_extractor.load_state_dict(weights['model_state_dict'])
        backbone = timm.create_model('tresnet_m.miil_in21k_ft_in1k',
                                     pretrained=True,
                                     num_classes=6)

        self.model = feat_extractor.encoder
        self.head = nn.Sequential(list(backbone.children())[-1])

    def forward(self, x):
        x_ = self.model(x)
        x_ = self.head(x_)

        return x_


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = PlantDataset(df, TRANSFORMER)
    x, y = next(iter(DataLoader(dataset, 4, True)))

    m = TResNet()
    main = m(x)

    print(main.shape)
