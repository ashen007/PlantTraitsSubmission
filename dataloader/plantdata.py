import warnings

warnings.filterwarnings('ignore')

import os.path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import albumentations as A
import cv2

from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class PlantTraitDataset(Dataset):
    TRANSFORMER = Compose([
        A.RandomResizedCrop(height=256, width=256),
        A.Flip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            # A.RandomGamma(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.Blur(p=0.1),
            A.GaussianBlur(p=0.1),
            A.MotionBlur(p=0.1),
        ], p=0.1),
        # A.OneOf([
        #     A.GaussNoise(p=0.1),
        #     A.ISONoise(p=0.1),
        #     A.GridDropout(ratio=0.5, p=0.2),
        #     A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
        # ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    def __init__(self, path, anno):
        self.dir = os.path.join(path, 'train_images')
        self.df = pd.read_csv(os.path.join(anno, 'train.csv'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        columns = self.df.columns

        img_id = self.df.loc[idx, 'id']
        y = torch.tensor(self.df.loc[idx, columns[1:]].values, dtype=torch.float32)

        img = cv2.imread(f'{self.dir}/{img_id}.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.TRANSFORMER(image=img)
        img = augmented['image']

        return img, y


if __name__ == '__main__':
    dataset = PlantTraitDataset('../data', '../data/processed')
    loader = DataLoader(dataset, 1, True)

    x, y = next(iter(loader))
    print(len(loader))
    print(x.shape)

    print(y)
    print(y.shape)

    plt.figure()
    plt.imshow(x[0].permute(1, 2, 0).numpy())
    plt.show()
