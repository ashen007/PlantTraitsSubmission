import numpy as np
import torch
import joblib
import pandas as pd
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
        self.scaling = joblib.load('../../data/processed/scaler_x.joblib')
        self.Xs = self.scaling.transform(x_features.loc[:, self.xs_cols[1:-2]])

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]))['image']
        xs = np.asarray(self.Xs[index, :])
        y_sample = self.y[index]

        return ((move_to(X_sample, 'cuda').unsqueeze(0),
                 move_to(torch.tensor(xs, dtype=torch.float32), 'cuda').unsqueeze(0)
                 ),
                y_sample)


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
