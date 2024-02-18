from torch import nn
from net_trainer import *
from models.approch_1.backbone.v2 import CustomConvNextSmall

if __name__ == "__main__":
    config = Config('../../../data/',
                    '../../../data/processed',
                    CustomConvNextSmall(),
                    nn.SmoothL1Loss,
                    20,
                    32,
                    restore_best=True,
                    multi_out=False)

    do(config)
