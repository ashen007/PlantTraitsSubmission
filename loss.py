import torch
from torch import nn


class R2Loss(nn.Module):

    def __init__(self):
        super(R2Loss, self).__init__()

        self.y_mean = torch.load('../../data/y_train_mean.pth')
        self.eps = torch.tensor([1e-6]).to('cuda')

    def forward(self, y_pred, y_true):
        rss = torch.sum((y_true - y_pred) ** 2, dim=0)
        tss = torch.sum((y_true - self.y_mean) ** 2, dim=0)
        eps = torch.full_like(tss, 1e-6)
        r2 = rss / (tss + eps)

        return torch.mean(r2)


class AltR2Loss(nn.Module):

    def __init__(self):
        super(AltR2Loss, self).__init__()

        self.y_mean = torch.load('../../data/y_train_mean.pth')
        self.eps = torch.tensor([1e-6]).to('cuda')

    def forward(self, y_pred, y_true):
        ess = torch.sum((y_pred - self.y_mean) ** 2, dim=0)
        tss = torch.sum((y_true - self.y_mean) ** 2, dim=0)
        eps = torch.full_like(tss, 1e-6)
        r2 = ess / (tss + eps)

        return torch.mean(r2)