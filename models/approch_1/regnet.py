import torch
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from torchvision.models.regnet import regnet_y_16gf
from models.approch_1.linear_models.recurrent import RecurrentTraitNet


class RegNet(nn.Module):

    def __init__(self, in_channels):
        super(RegNet, self).__init__()

        pre_trained_w = regnet_y_16gf(pretrained=True)
        self.feat_extractor = nn.Sequential(*(list(pre_trained_w.children())[:-1]))
        self.fc_branch = RecurrentTraitNet(489, 256, 256, 2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(3280, 1024),
                                        nn.LeakyReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 6))

    def forward(self, x):
        x1 = self.feat_extractor(x[0])
        x1 = self.flatten(x1)

        x2 = self.fc_branch(x[1])
        x_concat = torch.concat((x1, x2), dim=1)

        main = self.classifier(x_concat)
        aux = self.classifier(x_concat)

        return main, aux


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../data', '../data/processed')
    x, y = next(iter(DataLoader(dataset, 32, True)))
    m = RegNet(3)
    main, aux = m(x)

    loss = nn.MSELoss()
    loss_score = 0.5 * loss(main, y[0]) + 0.5 * loss(aux, y[1])

    print(main.shape, aux.shape)
    print(loss_score)
