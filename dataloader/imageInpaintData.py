import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from dataloader.transformers import INPAINT_TRANSFORMER


class ImageDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df
        self.img_size = 224
        self.mask_size = 64
        self.trans = transform

    def random_mask(self, image):
        y_1, x_1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y_2, x_2 = y_1 + self.mask_size, x_1 + self.mask_size

        masked_part = image[:, y_1: y_2, x_1: x_2]
        masked_img = image.clone()
        masked_img[:, y_1: y_2, x_1: x_2] = 1.0

        return masked_img, masked_part

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)
        Y = self.trans(image=image_file)['image']
        X, aux = self.random_mask(Y)

        return X, aux


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = ImageDataset(df, INPAINT_TRANSFORMER)

    x, aux = dataset[0]

    f, a = plt.subplots(1, 2)
    a[0].imshow(x.permute(1, 2, 0))
    a[1].imshow(aux.permute(1, 2, 0))
    plt.show()
