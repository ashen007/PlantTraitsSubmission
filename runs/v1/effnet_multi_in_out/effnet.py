from torch import nn
from net_trainer import *
from models.approch_1.backbone.effnet import CustomEffNet

if __name__ == "__main__":
    config = Config('../../data/',
                    '../../data/processed',
                    CustomEffNet(),
                    nn.MSELoss,
                    30,
                    16,
                    restore_best=True,
                    multi_out=True)

    do(config)
