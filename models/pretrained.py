import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from models.swin import SwinTransformer


class SwinTransformer_(nn.Module):
    def __init__(self):
        super(SwinTransformer_, self).__init__()

        # pre_trained_model = SwinTransformer()
        self.model = timm.create_model('swinv2_tiny_window16_256.ms_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)
        # x_ = self.fc(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    # print(x.shape, y.shape)

    m = SwinTransformer_()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
