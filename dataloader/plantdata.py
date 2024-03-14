import warnings

warnings.filterwarnings('ignore')

import os.path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from dataloader.transformers import TRANSFORMER


class PlantTraitDataset(Dataset):

    def __init__(self, path, transform=True):
        self.columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
        self.dir = os.path.join(path, 'train_images')
        self.df = pd.read_csv(os.path.join(path, 'processed/train.csv'))

        self.df['box'] = self.df['box'].apply(
            lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' ')
        )

        self.boxes = self.df.pop('box')

        self.xs = pd.read_csv(os.path.join(path, 'processed/train_x.csv')).drop('id', axis=1)
        self.xs_cols = self.xs.columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        x = self.xs.loc[idx, :]
        y = torch.tensor(self.df.loc[idx, self.columns].values, dtype=torch.float32)

        img = cv2.imread(f'{self.dir}/{img_id}.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box = self.boxes.loc[idx]
        img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        print(img.shape)

        if self.transform:
            augmented = TRANSFORMER(image=img)
            img = augmented['image']

        return ((img,
                 torch.tensor(x.values, dtype=torch.float32)),
                y)


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    loader = DataLoader(dataset, 1, True)

    (x1, x2), y = next(iter(loader))
    print(len(loader))
    print(x1.shape)
    print(x2.shape)

    print(x2)

    print(y)
    print(y.shape)

    plt.figure()
    plt.imshow(x1[0].permute(1, 2, 0).numpy())
    plt.show()
