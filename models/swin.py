import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()

        backbone = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                                     pretrained=True)

        self.model = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Sequential(*(list(list(backbone.children())[-1].children())[:-1]))
        self.out_layer = nn.Sequential(nn.Linear(1000, 6))


    def forward(self, x):
        x_ = self.model(x[0])
        x_ = self.fc(x_)
        x_ = self.out_layer(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    # print(x.shape, y.shape)

    m = SwinTransformer()
    main = m(x)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
