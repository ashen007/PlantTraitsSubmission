import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from torch.utils.data import Dataset, DataLoader
from dataloader.transformers import TEST_TRANSFORMER
from move import move_to


class TestDataset(Dataset):
    def __init__(self, X_jpeg_bytes, x_features, y):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.y = y
        self.transforms = TEST_TRANSFORMER
        self.xs_cols = x_features.columns
        # self.scaling = joblib.load('../../data/processed/scaler_x.joblib')
        # self.Xs = pd.DataFrame(self.scaling.transform(x_features.loc[:, self.xs_cols[1:-2]].values)).set_index(x_features['id'])
        self.boxes = pd.read_csv('../data/2024/boxes_test.csv', index_col='id')

        self.boxes['box'] = self.boxes['box'].apply(
            lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' ')
        )

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        try:
            box = self.boxes.loc[self.y[index], 'box']
            X_sample = self.transforms(
                image=imageio.imread(self.X_jpeg_bytes[index])[int(box[1]):int(box[3]), int(box[0]):int(box[2])])[
                'image']
        except:
            X_sample = self.transforms(
                image=imageio.imread(self.X_jpeg_bytes[index]))['image']

        # xs = np.asarray(self.Xs.loc[self.y[index], :])
        y_sample = self.y[index]

        return move_to(X_sample, 'cuda').unsqueeze(0), y_sample


if __name__ == '__main__':
    df = pd.read_pickle('../data/test.pkl')
    dataset = TestDataset(df['jpeg_bytes'].values,
                          df,
                          df['id'].values)
    loader = DataLoader(dataset, 1, True)

    (x1, x2), y = next(iter(loader))
    print(len(loader))
    print(x1.shape)
    print(x2.shape)
    print(y)
