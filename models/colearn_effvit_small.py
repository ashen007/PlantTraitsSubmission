import timm
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataloader.CLData import CLDataset
from dataloader.transformers import TRANSFORMER


class CoLearnEffnet(nn.Module):

    def __init__(self):
        super(CoLearnEffnet, self).__init__()

        backbone = timm.create_model('efficientvit_b2.r288_in1k', pretrained=True, num_classes=6)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))
        self.reg_head = nn.Sequential(nn.Linear(384, 6))
        self.proj_head = nn.Sequential(nn.Linear(384, 512, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 128))

    def forward(self, x, forward_fc=True, ignore_feat=False):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        features = torch.flatten(x, start_dim=1)
        projection = self.proj_head(features)

        if forward_fc:
            logits = self.reg_head(features)

            if ignore_feat:
                return projection, logits
            else:
                return features, projection, logits

        else:
            if ignore_feat:
                return projection
            else:
                return features, projection


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = CLDataset(df, TRANSFORMER)
    raw, img1, img2, y = next(iter(DataLoader(dataset, 4, True)))

    print(raw.shape, img1.shape, img2.shape)

    m = CoLearnEffnet()
    feat, proj, logit = m(img1)

    print(feat.shape, proj.shape, logit.shape)
