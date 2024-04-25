import scipy
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from scipy import special

eps = 1e-12


def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d


def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij


def CalPairwise_t(dist, v):
    C = scipy.special.gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * scipy.special.gamma(v / 2))
    return torch.pow((1 + torch.pow(dist, 2) / v), - (v + 1) / 2)


def loss_structrue(feat1, feat2):
    q1 = CalPairwise(Distance_squared(feat1, feat1))
    q2 = CalPairwise(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()


def loss_structrue_t(feat1, feat2, v):
    q1 = CalPairwise_t(Distance_squared(feat1, feat1), v)
    q2 = CalPairwise_t(Distance_squared(feat2, feat2), v)
    return -1 * (q1 * torch.log(q2 + eps)).mean()
