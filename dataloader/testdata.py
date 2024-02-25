import imageio.v3 as imageio
from torch.utils.data import Dataset
from dataloader.transformers import TEST_TRANSFORMER


class TestDataset(Dataset):
    def __init__(self, X_jpeg_bytes, y):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.y = y
        self.transforms = TEST_TRANSFORMER

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]))['image']
        y_sample = self.y[index]

        return X_sample, y_sample
