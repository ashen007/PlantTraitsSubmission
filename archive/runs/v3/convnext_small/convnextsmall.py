from torch import nn
from archive.net_trainer import *
from archive.models.approch_2.effnetlg import CustomConvNextSmall


if __name__ == "__main__":
    config = Config('../../../data/',
                    '../../../data/processed',
                    CustomConvNextSmall(),
                    nn.SmoothL1Loss,
                    25,
                    32,
                    restore_best=True,
                    multi_out=False)

    do(config)
