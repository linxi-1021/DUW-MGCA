import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAggregator(nn.Module):
    """Four-branch Gate + Aggregator.

    Implements the user's design:
      - Pool local QM/MD -> [B,H]
      - Use global QM/MD [B,H]
      - concat -> gate MLP -> softmax(4) -> scalar gates
      - weighted sum -> fused [B,H] -> optional proj -> optional predictor

    The pooling uses attention weights if provided (attn_qm2md for MD pooling via QM queries,
    attn_md2qm for QM pooling via MD queries). If not provided, falls back to mean pooling.
    """

    def __init__(self, fusion_hidden_dim, gate_hidden = None, proj_after = True, predictor = True, dropout = 0.0, eps = 1e-6, preserve_fused_scale: bool = False, fused_residual_scale: float = 0.0):
        super().__init__()
        self.fusion_hidden_dim = fusion_hidden_dim
        self.eps = eps
        gate_hidden = gate_hidden or max(128, self.fusion_hidden_dim)

        # gate network: (4H) -> gate_hidden -> 4 logits
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * self.fusion_hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(gate_hidden, 4),
        )

        # projection after fusion
        self.proj_after = proj_after
        if proj_after:
            self.after_proj = nn.Sequential(nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim), nn.ReLU())

        # optional predictor head
        self.predictor = predictor
        if predictor:
            self.pred_head = nn.Linear(self.fusion_hidden_dim, 1)
        # optional controls
        self.preserve_fused_scale = bool(preserve_fused_scale)
        self.fused_residual_scale = float(fused_residual_scale)
        if self.fused_residual_scale and self.fused_residual_scale > 0.0:
            self.residual_proj = nn.Linear(self.fusion_hidden_dim * 4, self.fusion_hidden_dim)

    def _pool_tokens(self, tokens, attn_from_other):
        """Pool tokens [B, L, H] -> [B, H]. Use attn_from_other [B, L_q, L] if available.

        If attn is provided but shape mismatches, fallback to mean pooling.
        attn_from_other is the attention from the other modality's queries to these tokens.
        For example, for pooling QM tokens, attn_from_other is attn_md2qm [B, L_md, L_qm].
        """
        if tokens is None:
            raise ValueError("tokens must be provided for pooling")

        if attn_from_other is None:
            return tokens.mean(dim=1)

        if attn_from_other.dim() != 3:
            return tokens.mean(dim=1)

        B, L, H = tokens.shape
        if attn_from_other.size(-1) != L:
            return tokens.mean(dim=1)

        importance = attn_from_other.mean(dim=1)  # [B, L] average attention over queries
        importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        denom = importance.sum(dim=1, keepdim=True).clamp(min=self.eps)
        importance = importance / denom
        pooled = torch.einsum("bl,blh->bh", importance, tokens)
        return pooled

    def forward(self, local_qm: torch.Tensor, local_md, global_qm, global_md, attn_qm2md = None, attn_md2qm = None):
        """Compute fused representation.

        Args:
            local_qm: [B, L_qm, H]
            local_md: [B, L_md, H]
            global_qm: [B, H] or None
            global_md: [B, H] or None
            attn_qm2md: [B, L_qm, L_md] (optional) -- used to pool md via QM queries
            attn_md2qm: [B, L_md, L_qm] (optional) -- used to pool qm via MD queries
        """
        if local_qm is None or local_md is None:
            raise ValueError("local_qm and local_md are required")

        # Sanitize inputs
        local_qm = torch.nan_to_num(local_qm, nan=0.0, posinf=0.0, neginf=0.0)
        local_md = torch.nan_to_num(local_md, nan=0.0, posinf=0.0, neginf=0.0)

        if local_qm.dim() != 3 or local_md.dim() != 3:
            raise ValueError("local_qm/local_md must be [B,L,H]")

        B, L_q, H_q = local_qm.shape
        B2, L_m, H_m = local_md.shape
        if B != B2:
            raise ValueError("Batch size mismatch between local_qm and local_md")

        if H_q != self.fusion_hidden_dim or H_m != self.fusion_hidden_dim:
            raise ValueError(f"Hidden dim mismatch: expected {self.fusion_hidden_dim}, got qm {H_q}, md {H_m}")

        # pool locals
        qm_local_pooled = self._pool_tokens(local_qm, attn_md2qm)
        md_local_pooled = self._pool_tokens(local_md, attn_qm2md)

        # globals: ensure shape [B,H]
        if global_qm is None:
            global_qm_in = qm_local_pooled
        else:
            if global_qm.dim() == 1:
                global_qm_in = global_qm.unsqueeze(0)
            else:
                global_qm_in = global_qm

        if global_md is None:
            global_md_in = md_local_pooled
        else:
            if global_md.dim() == 1:
                global_md_in = global_md.unsqueeze(0)
            else:
                global_md_in = global_md

        # concat four branches -> [B, 4H]
        concat = torch.cat([qm_local_pooled, md_local_pooled, global_qm_in, global_md_in], dim=-1)

        # gate logits and normalized gates
        gate_logits = self.gate_mlp(concat)  # [B,4]
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        gates = F.softmax(gate_logits, dim=-1)

        # apply scalar gates to each branch
        w0 = gates[:, 0:1]
        w1 = gates[:, 1:2]
        w2 = gates[:, 2:3]
        w3 = gates[:, 3:4]

        term0 = w0 * qm_local_pooled
        term1 = w1 * md_local_pooled
        term2 = w2 * global_qm_in
        term3 = w3 * global_md_in

        fused = term0 + term1 + term2 + term3

        if self.proj_after:
            fused = self.after_proj(fused)

        # optional fused scale preservation: match mean std of pooled branches
        fused_stat = {}
        if getattr(self, 'preserve_fused_scale', False):
            try:
                in_std_qm = qm_local_pooled.std(dim=1).mean()
                in_std_md = md_local_pooled.std(dim=1).mean()
                in_std_gqm = global_qm_in.std(dim=1).mean()
                in_std_gmd = global_md_in.std(dim=1).mean()
                in_std = float(((in_std_qm + in_std_md + in_std_gqm + in_std_gmd) / 4.0).detach().cpu().item())
                
                out_std_t = fused.std(dim=1).mean().detach()
                if not torch.isfinite(out_std_t):
                    out_std = 1.0
                else:
                    out_std = float(out_std_t.cpu().item())

                if out_std > 1e-8:
                    scale_val = max(min(in_std / (out_std + 1e-8), 10.0), 0.1)
                    scale = torch.tensor(scale_val, device=fused.device, dtype=fused.dtype)
                    fused = fused * scale
                    fused_stat['preserve_scale_in_std'] = in_std
                    fused_stat['preserve_scale_out_std'] = float(out_std)
                    fused_stat['preserve_scale_scale_used'] = float(scale_val)
            except Exception:
                pass

        # optional residual from concat to fused to reintroduce variance
        if getattr(self, 'fused_residual_scale', 0.0) and self.fused_residual_scale > 0.0:
            try:
                concat_flat = concat
                res = self.residual_proj(concat_flat)
                res = torch.clamp(res, -100.0, 100.0)
                fused = fused + self.fused_residual_scale * res
            except Exception:
                pass

        # sanitize fused to remove non-finite values (prevent downstream NaNs)
        if isinstance(fused, torch.Tensor) and not torch.isfinite(fused).all():
            bad_frac = (~torch.isfinite(fused)).float().mean().item()
            import logging
            logging.warning(f"[GatedAggregator] fused has non-finite values (fraction={bad_frac}); applying nan_to_num")
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        out = {
            "fused_representation": fused,
            "gates": gates,
            "gate_logits": gate_logits,
            "pooled": {
                "local_qm": qm_local_pooled,
                "local_md": md_local_pooled,
                "global_qm": global_qm_in,
                "global_md": global_md_in,
            },
        }

        if self.predictor:
            with torch.cuda.amp.autocast(enabled=False):
                out_pred = self.pred_head(fused.float())
            if isinstance(out_pred, torch.Tensor) and not torch.isfinite(out_pred).all():
                pf = (~torch.isfinite(out_pred)).float().mean().item()
                import logging
                logging.warning(f"[GatedAggregator] prediction tensor non-finite fraction={pf}; nan_to_num applied")
                out_pred = torch.nan_to_num(out_pred, nan=0.0, posinf=0.0, neginf=0.0)
            out["prediction"] = out_pred
        if fused_stat:
            out['fused_stats'] = fused_stat

        return out






