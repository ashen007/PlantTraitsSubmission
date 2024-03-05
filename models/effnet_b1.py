import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from models.dolg import *


class CustomEffnetVitB1(nn.Module):
    def __init__(self):
        super(CustomEffnetVitB1, self).__init__()

        self.model = timm.create_model('efficientvit_b1.r288_in1k',
                                       pretrained=True,
                                       features_only=True,
                                       out_indices=(2, 3))
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(int(2 * 256), 6)

    def forward(self, x):
        x_ = self.model(x[0])
        local_feat = self.local_branch(x_[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(x_[1]).squeeze())  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    # print(x.shape, y.shape)

    m = CustomEffnetVitB1()
    main = m(x)

    loss = nn.MSELoss()

    loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(loss_score)
