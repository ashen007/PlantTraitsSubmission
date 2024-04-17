import os.path

import matplotlib.pyplot as plt
import pandas as pd
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from dataloader.transformers import TEST_TRANSFORMER


class PlantDataset(Dataset):

    def __init__(self, df, transformers):
        self.df = df
        self.trans = transformers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)

        X = self.trans(image=image_file)['image']

        return X.unsqueeze(0), idx


if __name__ == '__main__':
    df = pd.read_csv('../data/test.csv')
    dataset = PlantDataset(df, TEST_TRANSFORMER)

    img, y = dataset[0]
    print(y)

    plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
