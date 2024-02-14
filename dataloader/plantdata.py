import warnings

warnings.filterwarnings('ignore')

import os.path
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torchvision import io
from torchvision.transforms.v2 import (Compose,
                                       CenterCrop,
                                       RandomCrop,
                                       RandomHorizontalFlip,
                                       ColorJitter,
                                       Normalize,
                                       RandomErasing,
                                       ToDtype
                                       )
from torch.utils.data import Dataset, DataLoader


class PlantTraitDataset(Dataset):
    TRANSFORMER = Compose([RandomCrop(320),
                           ColorJitter(brightness=.5, hue=.3, contrast=.5, saturation=.3),
                           RandomHorizontalFlip(),
                           ToDtype(torch.float32),
                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           RandomErasing()
                           ])

    def __init__(self, path, anno):
        self.dir = os.path.join(path, 'train_images')
        self.df = pd.read_csv(os.path.join(anno, 'train.csv'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        columns = self.df.columns

        img_id = self.df.loc[idx, 'id']
        xs = self.df.loc[idx, columns[1: -12]].values
        xs = torch.tensor(xs, dtype=torch.float32)

        ys_1 = self.df.loc[idx, columns[-12:-6]].values
        ys_1 = torch.tensor(ys_1, dtype=torch.float32)
        ys_2 = self.df.loc[idx, columns[-6:]].values
        ys_2 = torch.tensor(ys_2, dtype=torch.float32)

        img = torch.tensor(io.read_image(f'{self.dir}/{img_id}.jpeg'), dtype=torch.float32)/255.0
        img = self.TRANSFORMER(img)

        return (img, xs), (ys_1, ys_2)


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data', '../data/processed')
    loader = DataLoader(dataset, 1, True)

    xs, ys = next(iter(loader))

    print(xs[0].shape)
    print(xs[1].shape)
    print(ys[0].shape)
    print(ys[1].shape)

    print(xs[0][0])

    plt.figure()
    plt.imshow(xs[0][0].permute(1, 2, 0).numpy())
    plt.show()
