import warnings

warnings.filterwarnings('ignore')

import os.path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from dataloader.transformers import TRANSFORMER


class PlantTraitDataset(Dataset):

    def __init__(self, path):
        self.columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
        self.dir = os.path.join(path, 'train_images')
        self.df = pd.read_csv(os.path.join(path, 'processed/train.csv'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        y = torch.tensor(self.df.loc[idx, self.columns].values, dtype=torch.float32)

        img = cv2.imread(f'{self.dir}/{img_id}.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = TRANSFORMER(image=img)
        img = augmented['image']

        return img, y


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data')
    loader = DataLoader(dataset, 32, True)

    x, y = next(iter(loader))
    print(len(loader))
    print(x.shape)

    print(y)
    print(y.shape)

    plt.figure()
    plt.imshow(x[0].permute(1, 2, 0).numpy())
    plt.show()
