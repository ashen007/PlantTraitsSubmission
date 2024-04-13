import os.path

import matplotlib.pyplot as plt
import pandas as pd
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from dataloader.transformers import TRANSFORMER


class PlantDataset(Dataset):

    def __init__(self, df, transformers):
        self.df = df
        self.trans = transformers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)

        X = self.trans(image=image_file)['image']
        Y = list(self.df.iloc[idx, 1:].values)

        return X, torch.tensor(Y, dtype=torch.float32)


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = PlantDataset(df, TRANSFORMER)

    img, y = dataset[0]

    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
