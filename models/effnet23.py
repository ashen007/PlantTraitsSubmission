import timm
import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset2023
from torch.utils.data import DataLoader


class CustomEffnetFullY(nn.Module):
    def __init__(self):
        super(CustomEffnetFullY, self).__init__()

        backbone = timm.create_model('efficientvit_b2.r288_in1k',
                                     pretrained=True)

        self.model = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Sequential(*(list(list(backbone.children())[-1].children())[:-1]))
        self.out_layer = nn.Sequential(nn.Linear(2304, 30))

        # self.reg = nn.Sequential(nn.Linear(384, 1024),
        #                          nn.LayerNorm(1024),
        #                          nn.ReLU(),
        #                          nn.Dropout(0.5),
        #                          nn.Linear(1024, 1024),
        #                          nn.LayerNorm(1024),
        #                          nn.ReLU(),
        #                          nn.Dropout(0.4),
        #                          nn.Linear(1024, 6))

    def forward(self, x):
        x_ = self.model(x[0])
        x_ = self.fc(x_)
        x_ = self.out_layer(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset2023('../data/2023/')
    x1, y = next(iter(DataLoader(dataset, 1, True)))

    # print(x.shape, y.shape)

    m = CustomEffnetFullY()
    main = m(x1)

    # loss = nn.MSELoss()

    # loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    # print(loss_score)
