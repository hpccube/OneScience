# TODO(Dallas) Introduce Ensemble RMSE and MSE routines.

from typing import Union

import torch

Tensor = torch.Tensor


def mse(pred: Tensor, target: Tensor, dim: int = None) -> Union[Tensor, float]:
    """Calculates Mean Squared error between two tensors

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor
    target : Tensor
        Target tensor
    dim : int, optional
        Reduction dimension. When None the losses are averaged or summed over all
        observations, by default None

    Returns
    -------
    Union[Tensor, float]
        Mean squared error value(s)
    """
    return torch.mean((pred - target) ** 2, dim=dim)


def rmse(pred: Tensor, target: Tensor, dim: int = None) -> Union[Tensor, float]:
    """Calculates Root mean Squared error between two tensors

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor
    target : Tensor
        Target tensor
    dim : int, optional
        Reduction dimension. When None the losses are averaged or summed over all
        observations, by default None

    Returns
    -------
    Union[Tensor, float]
        Root mean squared error value(s)
    """
    return torch.sqrt(mse(pred, target, dim=dim))
