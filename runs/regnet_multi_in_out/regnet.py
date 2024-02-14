from torch import nn
from net_trainer import *
from models.regnet import RegNet

if __name__ == "__main__":
    config = Config('../../data/',
                    '../../data/processed',
                    RegNet(3),
                    nn.MSELoss,
                    30,
                    16,
                    restore_best=True,
                    multi_out=True)

    do(config)
