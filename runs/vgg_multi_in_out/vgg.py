from torch import nn
from net_trainer import *
from models.vgg import VGG

if __name__ == "__main__":
    config = Config('../../data/',
                    '../../data/processed',
                    VGG(3),
                    nn.MSELoss,
                    20,
                    32,
                    restore_best=True,
                    multi_out=True)

    do(config)
