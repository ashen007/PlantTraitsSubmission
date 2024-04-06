from torch.utils.data import DataLoader, random_split
from dataloader.plantdata import PlantTraitDataset, PlantTraitDataset2023
from dataloader.validdata import PlantTraitValidDataset, PlantTraitValDataset2023

DATASET = PlantTraitDataset('../../data/2024/')
train_size = int(len(DATASET) * 0.8)
test_size = len(DATASET) - train_size

# DATASET_VAL = PlantTraitValidDataset('../../data/2024/')
trainset, valset = random_split(DATASET, (train_size, test_size))


# DATASET_23 = PlantTraitDataset2023('../../data/2023/')
# DATASET_VAL_23 = PlantTraitValDataset2023('../../data/2023/')

# DATASET_FY_23 = PlantTraitDataset2023('../../data/2023/', full_y=True)
# DATASET_VAL_FY_23 = PlantTraitValDataset2023('../../data/2023/', full_y=True)


def get_train_dataloader(batch_size):
    return DataLoader(trainset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)


def get_val_dataloader(batch_size):
    return DataLoader(valset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True)

# def get_23_train_dataloader(batch_size, full_y=False):
#     if full_y:
#         return DataLoader(DATASET_FY_23,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           drop_last=True)

# else:
#     return DataLoader(DATASET_23,
#                       batch_size=batch_size,
#                       shuffle=True,
#                       drop_last=True)

# def get_23_val_dataloader(batch_size, full_y=False):
#     if full_y:
#         return DataLoader(DATASET_VAL_FY_23,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           drop_last=True)
#
#     else:
#         return DataLoader(DATASET_VAL_23,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           drop_last=True)
