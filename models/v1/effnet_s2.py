import torch
from torch import nn
from models.v1.effnet import CustomEffnet
from dataloader.plantdata import PlantTraitDataset2023
from torch.utils.data import DataLoader


class CustomEffnetS2(nn.Module):

    def __init__(self):
        super(CustomEffnetS2, self).__init__()

        backbone = CustomEffnet()
        state = torch.load('./step_5/best_checkpoint.pth')
        backbone.load_state_dict(state['model_state_dict'])

        self.model = nn.Sequential(*(list(backbone.children())[:-1]))
        self.out_layer = nn.Sequential(nn.Linear(2304, 30))

    def forward(self, x):
        x_ = self.model(x)
        x_ = self.out_layer(x_)

        return x_


if __name__ == '__main__':
    dataset = PlantTraitDataset2023('../../data/2023/')
    x, y = next(iter(DataLoader(dataset, 32, True)))

    print(x.shape, y.shape)

    m = CustomEffnetS2()
    main = m(x)

    loss = nn.MSELoss()

    loss_score = torch.sqrt(loss(main.cuda(), y.cuda()))

    print(main.shape)
    print(loss_score)
