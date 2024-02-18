from torch import nn
from net_trainer import *
from models.approch_1.exp import ExpNet

if __name__ == "__main__":
    config = Config('../../data/',
                    '../../data/processed',
                    ExpNet(3),
                    nn.MSELoss,
                    30,
                    32,
                    restore_best=True,
                    multi_out=True)

    do(config)
