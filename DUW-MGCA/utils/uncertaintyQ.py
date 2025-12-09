import torch
import torch.nn as nn
import torch.nn.functional as F

def _is_token_like(x):
    return (x is not None) and (x.dim() >= 2)

def _ensure_vector(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shape [B].
    Behavior:
      - If x is None -> None
      - If x is [B] or [B,1] -> return [B]
      - If x is [B, N] or [B, N, ...] -> return tensor unchanged (caller should aggregate with agg_uq)
    NOTE: we do NOT blindly mean-reduce high-dim tensors here anymore.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.dim() == 1:
            return x
        if x.dim() == 2 and x.size(1) == 1:
            return x.view(-1)
        # for higher dims, return as-is and let aggregator handle
        return x
    return x

def aggregate_token_uq(mean_tok: torch.Tensor, logvar_tok: torch.Tensor, eps=1e-8):
    """
    Aggregate token-level UQ into sample-level mean and logvar using law of total variance.

    Args:
      mean_tok: [B, N] or [B, N, 1] (or more trailing dims)  -> interpret per-token means
      logvar_tok: [B, N] or [B, N, 1] (same shape) -> per-token log variance (log var)
    Returns:
      agg_mean: [B]
      agg_logvar: [B]   (log of aggregated variance)
    Formula:
      agg_mean = mean_i m_i
      agg_var = mean_i (var_i + m_i^2) - agg_mean^2
      agg_logvar = log( agg_var + eps )
    """
    # flatten trailing dims
    if mean_tok is None:
        raise ValueError("mean_tok required")
    m = mean_tok
    if m.dim() > 2:
        m = m.view(m.size(0), m.size(1), -1).mean(dim=-1)  # [B,N]
    if logvar_tok is not None and logvar_tok.dim() > 2:
        lv = logvar_tok.view(logvar_tok.size(0), logvar_tok.size(1), -1).mean(dim=-1)
    else:
        lv = logvar_tok

    # ensure shapes [B,N]
    if m.dim() == 1:
        # already sample-level
        agg_mean = m
        if lv is None:
            agg_logvar = torch.zeros_like(agg_mean)
        else:
            agg_logvar = lv if lv.dim() == 1 else lv.mean(dim=1)
        return agg_mean, agg_logvar

    # now m is [B,N]
    # per-token variances
    if lv is None:
        # no per-token logvar -> treat per-token var as 0 (or small)
        var_tok = torch.zeros_like(m)
    else:
        var_tok = torch.exp(lv)

    # law of total variance
    mean_over_tokens = m.mean(dim=1)                 # [B]
    second_moment = (var_tok + m * m).mean(dim=1)    # E[var + m^2]
    agg_var = second_moment - mean_over_tokens * mean_over_tokens
    # numerical safety
    agg_var = torch.clamp(agg_var, min=1e-12)
    agg_logvar = torch.log(agg_var + eps)
    return mean_over_tokens, agg_logvar

class UncertaintyQuantification(nn.Module):
    def __init__(self, prefer_uq=True, clamp_logvar=(-10.0, 20.0), energy_agg="mean", eps=1e-8):
        super().__init__()
        self.prefer_uq = prefer_uq
        self.clamp_logvar = clamp_logvar
        self.energy_agg = energy_agg
        self.eps = eps

    def forward(self, ga_out):
        direct_raw = ga_out.get("prediction_direct", None)
        pred_field = ga_out.get("prediction", None)

        uq = ga_out.get("prediction_uq", None)
        if uq is None and pred_field is not None and not isinstance(pred_field, torch.Tensor):
            uq = pred_field

        # direct: explicit or tensor prediction
        direct = direct_raw if direct_raw is not None else (pred_field if isinstance(pred_field, torch.Tensor) else None)

        mean = None
        logvar = None

        # handle UQ formats
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

            # do not blindly reduce: keep token-level if present
            mean_c = _ensure_vector(mean_candidate)  # may be [B] or [B,N]
            logvar_c = _ensure_vector(logvar_candidate) if logvar_candidate is not None else None

            # if token-level (dim>=2): aggregate properly
            if isinstance(mean_c, torch.Tensor) and mean_c.dim() >= 2:
                # aggregate per-token UQ into sample-level
                mean, logvar = aggregate_token_uq(mean_c, logvar_c, eps=self.eps)
            else:
                # sample-level already
                mean = mean_c
                if logvar_c is not None:
                    if isinstance(logvar_c, torch.Tensor) and logvar_c.dim() >= 2:
                        # reduce token-level logvar to sample-level (mean)
                        logvar = logvar_c.view(logvar_c.size(0), -1).mean(dim=1)
                    else:
                        logvar = logvar_c

        # fallback to direct if mean missing
        if mean is None and direct is not None:
            mean = _ensure_vector(direct)
            if isinstance(mean, torch.Tensor) and mean.dim() >= 2:
                # reduce direct token-level to sample-level by mean (no var info)
                mean = mean.view(mean.size(0), -1).mean(dim=1)

        if mean is None:
            raise ValueError("No prediction found in ga_out (neither prediction_direct nor prediction_uq present).")

        # prepare logvar
        uq_active = logvar is not None
        if not uq_active:
            # if no logvar information, set to a conservative large value so UQ doesn't dominate
            logvar = torch.zeros_like(mean) + 1.0  # var=exp(1)=2.718 -> prevents UQ domination

        # clamp logvar
        if self.clamp_logvar is not None:
            lo, hi = self.clamp_logvar
            logvar = logvar.clamp(min=lo, max=hi)

        ga_out["prediction_mean"] = mean
        ga_out["prediction_logvar"] = logvar
        ga_out["prediction"] = ga_out.get("prediction", mean)
        ga_out["uq_active"] = uq_active

        return ga_out
