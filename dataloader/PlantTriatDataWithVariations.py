import os.path
import numpy as np
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
        self.names = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']
        self.y_means = [f'{y}_mean' for y in self.names]
        self.y_std = [f'{y}_sd' for y in self.names]
        self.epsilon_range = (-0.01, 0.01)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)
        alpha = np.random.uniform(self.epsilon_range[0],
                                  self.epsilon_range[1])

        Ys = self.df.loc[idx, self.y_means].values + alpha * self.df.loc[idx, self.y_std].values
        Ys = list(Ys)

        X = self.trans(image=image_file)['image']

        return X, torch.tensor(Ys, dtype=torch.float32)


if __name__ == '__main__':
    df = pd.read_csv('../data/train_with_sd.csv')
    dataset = PlantDataset(df, TRANSFORMER)

    img, y = dataset[0]

    print(y)

    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
