import timm
from torch import nn
from dataloader.plantdata import PlantTraitDataset
from torch.utils.data import DataLoader


class CustomConvNextSmall(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True):
        super(CustomConvNextSmall, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features

        self.feat_extractor = nn.Sequential(*(list(self.model.children())[:-1]))
        self.bckbone_head = nn.Sequential(*(list(list(self.model.children())[-1].children())[:-1]))
        self.model = nn.Sequential(self.feat_extractor,
                                   self.bckbone_head)
        self.classifier = nn.Sequential(nn.Linear(in_features, in_features),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features, 6)
                                        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    dataset = PlantTraitDataset('../../../data', '../../data/processed')
    x, y = next(iter(DataLoader(dataset, 16, True)))

    print(x.shape, y.shape)

    m = CustomConvNextSmall('convnext_small.fb_in22k')
    main = m(x)

    loss = nn.SmoothL1Loss()
    loss_score = loss(main, y)

    print(main.shape)
    print(loss_score)
