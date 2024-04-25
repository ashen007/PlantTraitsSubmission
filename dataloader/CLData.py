import pandas as pd
import albumentations as A
import imageio.v3 as imageio
import torch
import matplotlib.pyplot as plt
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from dataloader.transformers import TRANSFORMER


class CLDataset(Dataset):

    def __init__(self, df, transformers):
        self.df = df
        self.trans = transformers
        self.strong_trans = Compose([A.RandomResizedCrop(224, 224),
                                     A.HorizontalFlip(),
                                     A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                                     A.ToGray(p=0.2),
                                     A.ToFloat(),
                                     A.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225],
                                                 max_pixel_value=1),
                                     ToTensorV2()
                                     ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        path = f"../data{self.df.loc[idx, 'path']}"
        image_file = imageio.imread(path)

        X = self.trans(image=image_file)['image']
        img1 = self.strong_trans(image=image_file)['image']
        img2 = self.strong_trans(image=image_file)['image']
        Y = list(self.df.iloc[idx, 1:].values)

        return X, img1, img2, torch.tensor(Y, dtype=torch.float32)


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dataset = CLDataset(df, TRANSFORMER)

    img0, img1, img2, y = dataset[0]

    print(y)

    f, a = plt.subplots(1, 3)
    a[0].imshow(img0.permute(1, 2, 0))
    a[1].imshow(img1.permute(1, 2, 0))
    a[2].imshow(img2.permute(1, 2, 0))
    plt.show()
