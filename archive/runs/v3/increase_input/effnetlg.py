from torch import nn
from archive.net_trainer import *
from archive.models.approch_2.effnetlg import CustomEffnetLarge

if __name__ == "__main__":
    # retrain increased
    model = CustomEffnetLarge()

    config = Config('../../../data/',
                    '../../../data/processed',
                    model,
                    nn.MSELoss,
                    10,
                    32,
                    restore_best=True,
                    multi_out=False)

    do(config)
