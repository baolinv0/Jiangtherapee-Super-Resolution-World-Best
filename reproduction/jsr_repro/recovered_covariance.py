"""Steering covariance recovered from the official v9.8 CPU extension.

The input is the already constructed reference guide. RAW-to-guide interpolation
and registration remain outside this module. Outputs use the native guide grid;
the released fusion path consumes covariance on its output grid.

Frozen numeric probes cover the released extension SHA-256
20acf56aa47dd483759bf43b2d948a8bf1459936050b65f3fcd96737bafedbe4.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .frontend import reflect_pad, safe_sqrt

EPS = 1e-12
COVARIANCE_VERSION = 'release-v98-steering-covariance-v1'


def _validate(value: Tensor, channels: int, name: str) -> None:
    if (value.ndim != 4 or value.shape[1] != channels or min(value.shape) < 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite floating [B,{channels},H,W]')


def structure_tensor(guide: Tensor) -> Tensor:
    """Released Sobel*0.25, then a 3x3 unnormalized REFLECT_101 box filter.

    Returns channels (Jxx, Jxy, Jyy) on the supplied native guide grid.
    """
    _validate(guide, 1, 'guide')
    sobel = guide.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    padded = reflect_pad(guide, 1)
    gx = F.conv2d(padded, sobel[None, None]) * .25
    gy = F.conv2d(padded, sobel.t()[None, None]) * .25
    products = torch.cat((gx.square(), gx * gy, gy.square()), dim=1)
    box = guide.new_ones(3, 1, 3, 3)
    return F.conv2d(reflect_pad(products, 1), box, groups=3)


def covariance_from_tensor(
    tensor: Tensor, *, k_detail: float = .12, min_k: float = .2,
    k_stretch: float = 4., k_shrink: float = 2.,
) -> Tensor:
    """Convert (Jxx,Jxy,Jyy) into inverse covariance (xx,xy,yy).

    The released entry rejects changes to MIN_K=.2 and K_STRETCH=4.
    Stable eigenvector selection and flat/isotropic fallbacks follow controlled
    probes of that entry, including its asymmetric zero-tensor orientation.
    """
    _validate(tensor, 3, 'tensor')
    parameters = (k_detail, min_k, k_stretch, k_shrink)
    if (any(not math.isfinite(float(value)) or float(value) <= 0 for value in parameters)
            or min_k != .2 or k_stretch != 4.):
        raise ValueError('Positive finite parameters with MIN_K=.2 and K_STRETCH=4 required')
    jxx, jxy, jyy = tensor.unbind(dim=1)
    trace = jxx + jyy
    determinant = jxx * jyy - jxy.square()
    half_gap = safe_sqrt(trace.square() * .25 - determinant)
    largest = (trace * .5 + half_gap).clamp_min(EPS)
    smallest = (trace * .5 - half_gap).clamp_min(0.)
    anisotropy = safe_sqrt((largest - smallest) / (largest + smallest + EPS))
    blend = (1. + anisotropy) * .5
    sigma_along = k_detail * (1. + (k_stretch - 1.) * blend)
    sigma_across = (k_detail * (1. + (1. / k_shrink - 1.) * blend)).clamp_min(min_k)

    ax, ay = jxy, largest - jxx
    bx, by = largest - jyy, jxy
    pick_first = ax.square() + ay.square() >= bx.square() + by.square()
    ex = torch.where(pick_first, ax, bx)
    ey = torch.where(pick_first, ay, by)
    norm = safe_sqrt(ex.square() + ey.square())
    good = norm > 0
    denominator = torch.where(good, norm, torch.ones_like(norm))
    ex = torch.where(good, ex / denominator, torch.ones_like(ex))
    ey = torch.where(good, ey / denominator, torch.zeros_like(ey))
    inverse_along = 1. / sigma_along.square()
    inverse_across = 1. / sigma_across.square()
    difference = inverse_across - inverse_along
    return torch.stack((inverse_along + difference * ex * ex,
                        difference * ex * ey,
                        inverse_along + difference * ey * ey), dim=1)


def steering_covariance(
    guide: Tensor, *, k_detail: float = .12, min_k: float = .2,
    k_stretch: float = 4., k_shrink: float = 2.,
) -> Tensor:
    """Compute native-grid inverse covariance from an externally supplied guide."""
    return covariance_from_tensor(structure_tensor(guide), k_detail=k_detail,
                                  min_k=min_k, k_stretch=k_stretch, k_shrink=k_shrink)
