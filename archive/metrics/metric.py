import numpy as np
from sklearn.metrics import r2_score


def mean_r2(y_true, y_pred):
    scores = []

    print(y_true)

    for _ in range(6):
        scores.append(r2_score(y_true[:, _].detach().numpy(), y_pred[:, _].detach().numpy()))

    return tuple(scores)
