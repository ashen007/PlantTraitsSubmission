import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()

        self.model = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                                       pretrained=True,
                                       num_classes=6)

    def forward(self, x):
        x_ = self.model(x)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../data/2024/')
    x, y = next(iter(DataLoader(dataset, 4, True)))

    # print(x.shape, y.shape)

    m = SwinTransformer()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
