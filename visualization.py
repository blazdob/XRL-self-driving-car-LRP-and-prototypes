import numpy as np
import torch
import matplotlib.pyplot as plt

def project(X, output_range=(0, 1)):
    absmax   = np.abs(X).max()
    X       /= absmax + (absmax == 0).astype(float)
    X        = (X+1) / 2. # range [0, 1]
    X        = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
    return X


def clip_quantile(X, quantile=1):
    """Clip the values of X into the given quantile."""
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X


