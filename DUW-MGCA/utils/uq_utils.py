"""Utility helpers for uncertainty-aware losses and metrics.

Provides free functions that accept either a canonical ga_out dict (with
prediction_mean/logvar) or raw tensors. Other modules can import these
functions to compute RMSE, NLL, sampling and calibration metrics.
"""
from typing import Optional, Dict, Tuple, Union
import math

import torch
import torch.nn.functional as F


def _ensure_vector(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2 and x.size(1) == 1:
        return x.view(-1)
    if x.dim() == 1:
        return x
    return x.view(x.size(0), -1).mean(dim=1)


def to_canonical(
    ga_out: Dict,
    clamp_logvar: Optional[Tuple[float, float]] = (-20.0, 20.0),
    eps: float = 1e-8,
    device: Optional[torch.device] = None,
) -> Dict:
    """Return a canonical dict with keys prediction_mean, prediction_logvar,
    prediction and uq_active.

    ga_out may be any mapping that contains prediction_direct (tensor),
    prediction (tensor or tuple/dict), or prediction_uq (tuple/dict/tensor).
    """
    direct_raw = ga_out.get("prediction_direct", None)
    pred_field = ga_out.get("prediction", None)

    uq = ga_out.get("prediction_uq", None)
    if uq is None and pred_field is not None and not isinstance(pred_field, torch.Tensor):
        uq = pred_field

    direct = direct_raw if direct_raw is not None else (pred_field if isinstance(pred_field, torch.Tensor) else None)

    mean = None
    logvar = None

    if uq is not None:
        if isinstance(uq, (tuple, list)) and len(uq) >= 1:
            mean_candidate = uq[0]
            logvar_candidate = uq[1] if len(uq) > 1 else None
        elif isinstance(uq, dict):
            mean_candidate = uq.get("mean", uq.get("mu", None))
            logvar_candidate = uq.get("logvar", uq.get("var", None))
        else:
            mean_candidate = uq
            logvar_candidate = None

        mean = _ensure_vector(mean_candidate) if mean_candidate is not None else None
        if logvar_candidate is not None:
            logvar = _ensure_vector(logvar_candidate)

    if mean is None and direct is not None:
        mean = _ensure_vector(direct)

    if mean is None:
        raise ValueError("No prediction found in ga_out (neither prediction_direct nor prediction_uq present).")

    uq_active = logvar is not None
    if not uq_active:
        if device is None and mean is not None:
            device = mean.device
        logvar = torch.zeros_like(mean, device=device) if mean is not None else torch.zeros(0, device=device)

    if clamp_logvar is not None:
        lo, hi = clamp_logvar
        logvar = logvar.clamp(min=lo, max=hi)

    out = {
        "prediction_mean": mean,
        "prediction_logvar": logvar,
        "prediction": ga_out.get("prediction", mean),
        "uq_active": uq_active,
    }
    return out


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    # ensure tensors align on device
    device = mean.device if mean is not None else (target.device if target is not None else None)
    if device is not None:
        mean = mean.to(device)
        logvar = logvar.to(device)
        target = target.to(device)
    var = torch.exp(logvar)
    se = (target - mean) ** 2
    elementwise = 0.5 * (logvar + se / (var + eps)) + 0.5 * math.log(2 * math.pi)
    if reduction == "mean":
        return elementwise.mean()
    elif reduction == "sum":
        return elementwise.sum()
    else:
        return elementwise


def mse(mean: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(mean, target, reduction=reduction)


def uq_loss(
    ga_or_mean: Union[Dict, torch.Tensor],
    target: torch.Tensor,
    loss_type: str = "nll",
    reduction: str = "mean",
    clamp_logvar: Optional[Tuple[float, float]] = (-20.0, 20.0),
    eps: float = 1e-8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """High-level loss: accepts either a canonical ga_out dict or a mean tensor.

    If ga_or_mean is a dict, it will be standardized via to_canonical().
    """
    if isinstance(ga_or_mean, dict):
        canon = to_canonical(ga_or_mean, clamp_logvar=clamp_logvar, eps=eps, device=device)
        mean = canon["prediction_mean"]
        logvar = canon.get("prediction_logvar", None)
    else:
        mean = _ensure_vector(ga_or_mean)
        logvar = None

    if loss_type == "nll":
        if logvar is None:
            return mse(mean, target, reduction=reduction)
        return gaussian_nll(mean, logvar, target, eps=eps, reduction=reduction)
    elif loss_type == "mse":
        return mse(mean, target, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def sample_from_gaussian(ga_or_mean: Union[Dict, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
    if isinstance(ga_or_mean, dict):
        mean = ga_or_mean.get("prediction_mean")
        logvar = ga_or_mean.get("prediction_logvar")
    else:
        mean = _ensure_vector(ga_or_mean)
        logvar = None
    if mean is None or logvar is None:
        raise ValueError("Need prediction_mean and prediction_logvar to sample")
    std = torch.exp(0.5 * logvar)
    device = mean.device
    samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn((n_samples, mean.size(0)), device=device, dtype=mean.dtype)
    return samples


def calibration_metrics(ga_or_mean: Union[Dict, torch.Tensor], target: torch.Tensor, device: Optional[torch.device] = None) -> Dict[str, float]:
    if isinstance(ga_or_mean, dict):
        mean = ga_or_mean.get("prediction_mean")
        logvar = ga_or_mean.get("prediction_logvar")
    else:
        mean = _ensure_vector(ga_or_mean)
        logvar = None
    if mean is None:
        raise ValueError("Need prediction_mean in input to compute metrics")
    # move target to mean's device unless device explicitly provided
    target_dev = device or mean.device
    try:
        target = target.to(target_dev)
    except Exception:
        pass
    with torch.no_grad():
        rmse = torch.sqrt(F.mse_loss(mean, target, reduction="mean"))
        if logvar is not None:
            nll = gaussian_nll(mean, logvar, target, reduction="mean")
            sharpness = torch.exp(logvar).mean()
        else:
            nll = float('nan')
            sharpness = float('nan')
    return {"rmse": float(rmse.item()), 
            "nll": float(nll.item()) if isinstance(nll, torch.Tensor) else float(nll), 
            "sharpness": float(sharpness)}




__all__ = [
    "to_canonical",
    "gaussian_nll",
    "mse",
    "uq_loss",
    "sample_from_gaussian",
    "calibration_metrics",
]
