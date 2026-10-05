"""Linear, fixed-range metrics without fitted exposure/color transformations."""
from __future__ import annotations

import math
import torch


def crop_pair(prediction, target, border=4):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("metric inputs must be matching B,3,H,W tensors")
    if border < 0 or 2 * border >= min(prediction.shape[-2:]):
        raise ValueError("invalid metric crop border")
    if border:
        prediction = prediction[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("metric inputs contain NaN/Inf")
    return prediction.double(), target.double()


def image_metrics(prediction, target, border=4):
    prediction, target = crop_pair(prediction, target, border)
    error = prediction - target
    mse = error.square().mean().item()
    chroma = torch.stack((error[:, 0] - error[:, 1], error[:, 2] - error[:, 1]), 1)
    shadow = target.mean(1, keepdim=True) < .02
    shadow = shadow.expand_as(error)
    return {
        # Exact equality uses a documented finite 120 dB cap for strict JSON.
        "psnr_linear_db": -10 * math.log10(max(mse, 1e-12)),
        "rmse_linear": math.sqrt(mse),
        "chroma_rmse": chroma.square().mean().sqrt().item(),
        "shadow_rmse": error[shadow].square().mean().sqrt().item() if shadow.any() else None,
        "mean_bias": error.mean().item(),
    }


def reconstruction_loss(prediction, target, border=4, chroma_weight=.1, gradient_weight=.05):
    if border:
        prediction = prediction[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    if min(target.shape[-2:]) < 2:
        raise ValueError("loss crop leaves too little image")
    diff = prediction - target
    rgb = diff.abs().mean()
    chroma = ((diff[:, 0] - diff[:, 1]).abs().mean() + (diff[:, 2] - diff[:, 1]).abs().mean()) * .5
    gradient = ((diff[..., 1:, :] - diff[..., :-1, :]).abs().mean() + (diff[..., :, 1:] - diff[..., :, :-1]).abs().mean()) * .5
    return rgb + chroma_weight * chroma + gradient_weight * gradient
