from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, OneCycleLR
from torch.utils.data import random_split, DataLoader
from dataloader.plantdata import PlantTraitDataset
from train.train import Compile
from metrics.metric import mean_r2


@dataclass
class Config:

    def __init__(self,
                 root,
                 anno,
                 model,
                 loss,
                 epochs,
                 batch_size,
                 multi_in=False,
                 restore_best=True,
                 multi_out=False):
        # dataloaders
        self.data_root = root
        self.anno = anno

        # models
        self.lr = 1e-4
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.01) # only for tuning
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.15, patience=5, cooldown=5)
        self.scheduler = OneCycleLR(self.optimizer,
                                    epochs=epochs,
                                    steps_per_epoch=2559,
                                    max_lr=1e-3,
                                    pct_start=0.2,
                                    div_factor=1.0e+3,
                                    final_div_factor=1.0e+3)
        self.multi_in = multi_in
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.restore_best = restore_best
        self.multi_out = multi_out


def do(config: Config):
    # create dataset
    data_set = PlantTraitDataset(config.data_root, config.anno)
    print(f"Dataset: {data_set.__class__.__name__}")
    print(f"train image count: {len(data_set)}")

    # dataloaders
    train_size = int(len(data_set) * 0.8)
    test_size = len(data_set) - train_size
    train_dataset, test_dataset = random_split(data_set, (train_size, test_size))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=True)

    print(f"training steps: {len(train_dataloader)}  "
          f"validation steps: {len(test_dataloader)}")

    # model creation
    compiled = Compile(config.model,
                       config.optimizer,
                       lr_scheduler=config.scheduler,
                       # metrics={'mean_r2': mean_r2}
                       )

    results = compiled.train(train_dataloader,
                             config.loss(),
                             config.epochs,
                             test_dataloader,
                             multi_in=config.multi_in,
                             multi_out=config.multi_out)

    # save results
    results.to_csv('./results.csv', index=False)
