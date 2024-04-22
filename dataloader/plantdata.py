import warnings

warnings.filterwarnings('ignore')

import os.path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import cv2
from torch.utils.data import Dataset, DataLoader
from dataloader.transformers import TRANSFORMER


class PlantTraitDataset(Dataset):

    def __init__(self, df, transform):
        self.columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
        self.df = df

        self.df['box'] = self.df['box'].apply(
            lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' ')
        )

        self.boxes = self.df.pop('box')
        self.trans = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)
        box = self.boxes.loc[idx]
        img = image_file[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        X = self.trans(image=img)['image']
        Y = list(self.df.loc[idx, self.columns].values)

        return X, torch.tensor(Y, dtype=torch.float32)


class PlantTraitDataset2023(Dataset):

    def __init__(self, path, transform=True, full_y=False):
        self.dir = os.path.join(path, '01_data_train')
        self.df = pd.read_csv(os.path.join(path, 'processed/train.csv'))
        self.pic_names = self.df.pop('pic_name')
        self.transform = transform
        self.full_y = full_y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.full_y:
            y = torch.tensor(self.df.loc[idx, :].values[:6], dtype=torch.float32)

        else:
            y = torch.tensor(self.df.loc[idx, :].values, dtype=torch.float32)

        img = cv2.imread(f'{self.dir}/{self.pic_names[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = TRANSFORMER(image=img)
            img = augmented['image']

        return (img, y), y


if __name__ == '__main__':
    df = pd.read_csv('../data/train_with_boxes.csv')
    dataset = PlantTraitDataset(df, TRANSFORMER)
    # dataset_23 = PlantTraitDataset2023('../data/2023/', full_y=True)
    loader = DataLoader(dataset, 1, True)
    # loader = DataLoader(dataset_23, 1, True)

    # (x1, x2), y = next(iter(loader))
    # (x1, x2), y = next(iter(loader))
    x, y = next(iter(loader))

    # print(len(loader))
    # print(x1.shape)
    # print(x2.shape)

    print(x.shape)
    print(y.shape)

    # print(x2)
    # print(y)
    # print(y.shape)

    plt.figure()
    plt.imshow(x[0].permute(1, 2, 0).numpy())
    plt.show()
