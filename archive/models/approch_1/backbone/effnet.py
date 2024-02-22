import timm
import torch
from torch import nn
from torch.nn import functional as F
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from archive.models.approch_1.linear_models.recurrent import RecurrentTraitNet


class CustomEffNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        self.lstm_branch = RecurrentTraitNet(489, 256, 256)

        self.ch_att = nn.Sequential(nn.LSTM(in_features + 256, in_features + 256, 1,
                                            batch_first=True, bidirectional=False))
        self.classifier = nn.Sequential(nn.Linear(in_features + 256, in_features),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features, 6)
                                        )

    def forward(self, x):
        x1 = self.model(x[0])
        x2 = self.lstm_branch(x[1])

        x_cat = torch.cat((x1, x2), dim=1)
        x_cat_out, _ = self.ch_att(x_cat.view(len(x_cat), 1, -1))
        x_cat_score = F.sigmoid(x_cat_out.view(len(x_cat_out), -1))

        main = self.classifier(x_cat * x_cat_score)
        aux = self.classifier(x_cat * x_cat_score)

        return main, aux


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../../../data', '../../data/processed')
    x, y = next(iter(DataLoader(dataset, 32, True)))
    m = CustomEffNet()
    main, aux = m(x)

    loss = nn.MSELoss()
    loss_score = torch.sqrt(loss(main, y[0])) + torch.sqrt(loss(aux, y[1]))

    print(main.shape, aux.shape)
    print(loss_score)
