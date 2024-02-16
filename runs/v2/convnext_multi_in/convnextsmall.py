from torch import nn
from net_trainer import *
from models.backbone.v2.convnext_samle import CustomConvNextSmall

if __name__ == "__main__":
    config = Config('../../../data/',
                    '../../../data/processed',
                    CustomConvNextSmall(),
                    nn.SmoothL1Loss,
                    30,
                    32,
                    restore_best=True,
                    multi_out=False)

    do(config)
