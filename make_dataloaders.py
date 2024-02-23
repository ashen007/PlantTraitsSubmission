from torch.utils.data import DataLoader, random_split
from dataloader.plantdata import PlantTraitDataset

DATASET = PlantTraitDataset('../../data/',
                            '../../data/processed')

# dataloaders
train_size = int(len(DATASET) * 0.8)
test_size = len(DATASET) - train_size
train_dataset, test_dataset = random_split(DATASET, (train_size, test_size))


def get_train_dataloader(batch_size):
    return DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)


def get_val_dataloader(batch_size):
    return DataLoader(test_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)
