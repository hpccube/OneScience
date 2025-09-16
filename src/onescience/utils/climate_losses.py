import math
from typing import Optional

import torch
import torch.nn as nn
import torch.fft


def _latitude_weights(height: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Compute latitude weights proportional to cos(lat).

    Args:
        height: Number of latitude grid points (from -90 to 90).
        device: Torch device.
        dtype: Torch dtype.
    Returns:
        Tensor of shape [1, 1, H, 1] suitable for broadcasting over [B, C, H, W].
    """
    latitudes = torch.linspace(-90.0, 90.0, steps=height, device=device, dtype=dtype)
    weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp(min=0.0)
    # Normalize to mean 1.0 to keep loss scale comparable to vanilla MSE
    weights_1d = weights_1d / (weights_1d.mean() + 1e-6)
    return weights_1d.view(1, 1, height, 1)


def latitude_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Latitude-weighted mean squared error for global fields.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        Scalar tensor loss.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    _, _, h, _ = pred.shape
    weights = _latitude_weights(h, pred.device, pred.dtype)
    sq_err = (pred - target) ** 2
    weighted = sq_err * weights
    return weighted.mean()


def spectral_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    high_k_boost: float = 0.0,
) -> torch.Tensor:
    """MSE computed in Fourier domain with optional emphasis on high frequencies.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        high_k_boost: Extra weight applied linearly with radial frequency (0 = no boost).
    Returns:
        Scalar tensor loss.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    # Real FFT over spatial dims
    pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    targ_f = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
    diff = pred_f - targ_f
    power = (diff.real ** 2 + diff.imag ** 2)

    if high_k_boost > 0.0:
        b, c, h, w_r = power.shape
        # radial frequency grid
        ky = torch.fft.fftfreq(h, d=1.0, device=power.device).abs().view(1, 1, h, 1)
        kx = torch.fft.rfftfreq(h, d=1.0, device=power.device)  # use h to keep aspect ~1
        kx = kx.abs().view(1, 1, 1, w_r)
        kr = torch.sqrt(kx * kx + ky * ky)
        kr = kr / (kr.max() + 1e-6)
        weight = 1.0 + high_k_boost * kr
        power = power * weight

    return power.mean()


