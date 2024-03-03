from torch.utils.data import DataLoader, random_split
from dataloader.plantdata import PlantTraitDataset
from dataloader.validdata import PlantTraitValidDataset

DATASET = PlantTraitDataset('../../data/')
DATASET_VAL = PlantTraitValidDataset('../../data/')


def get_train_dataloader(batch_size):
    return DataLoader(DATASET,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)


def get_val_dataloader(batch_size):
    return DataLoader(DATASET_VAL,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)
