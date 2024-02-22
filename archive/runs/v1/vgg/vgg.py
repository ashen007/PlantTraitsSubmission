from torch import nn
from archive.net_trainer import *
from archive.models.approch_1.vgg import VGG

if __name__ == "__main__":
    config = Config('../../data/',
                    '../../data/processed',
                    VGG(3),
                    nn.MSELoss,
                    10,
                    32,
                    restore_best=True)

    do(config)
