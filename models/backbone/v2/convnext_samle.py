import timm
import torch
from torch import nn
from torch.nn import functional as F
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader
from models.linear_models.recurrent import RecurrentTraitNet


class CustomConvNextSmall(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super(CustomConvNextSmall, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.feat_extractor = nn.Sequential(*(list(self.model.children())[:-1]))
        self.bckbone_head = nn.Sequential(*(list(list(self.model.children())[-1].children())[:-1]))

        self.model = nn.Sequential(self.feat_extractor,
                                   self.bckbone_head)
        self.lstm_branch = RecurrentTraitNet(163, 128, 256)
        self.classifier = nn.Sequential(nn.Linear(in_features + 256, in_features),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features, 6)
                                        )

    def forward(self, x):
        x1 = self.model(x[0])
        x2 = self.lstm_branch(x[1])

        x_cat = torch.cat((x1, x2), dim=1)

        main = self.classifier(x_cat)

        return main


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../../data', '../../../data/processed')
    x, y = next(iter(DataLoader(dataset, 32, True)))
    m = CustomConvNextSmall('convnext_small.fb_in22k')
    main = m(x)

    loss = nn.SmoothL1Loss()
    loss_score = loss(main, y)

    print(main.shape)
    print(loss_score)
