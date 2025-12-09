import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """Small head that predicts mean and log-variance for UQ outputs."""
    def __init__(self, in_dim, hidden=128, out_dim=1, dropout=0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.mean = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mean(h), self.logvar(h)


class PredictionHead(nn.Module):
    """Prediction head that can emit direct outputs and UQ outputs.

    Params:
      hidden_dim: input feature dim (from GatedAggregator)
      token_direct_dim: number of token dims produced directly
      token_uq_dim: number of token dims produced by UQ head (returns mean, logvar)
      global_direct_dim: direct global output dims
      global_uq_dim: global dims produced by UQ head
    Returns:
      token_direct: Tensor or None (B,N,token_direct_dim)
      global_direct: Tensor or None (B,global_direct_dim)
      token_uq: (mean, logvar) or None where mean/logvar are (B,N,token_uq_dim)
      global_uq: (mean, logvar) or None where mean/logvar are (B,global_uq_dim)
    """

    def __init__(self, hidden_dim, token_direct_dim=0, token_uq_dim=0,
                 global_direct_dim=0, global_uq_dim=0, hidden=128, dropout=0.0, pool='max'):
        super().__init__()
        self.pool = pool

        # token direct MLP
        self.token_direct_dim = token_direct_dim
        if token_direct_dim > 0:
            self.token_direct_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, token_direct_dim)
            )
        else:
            self.token_direct_mlp = None

        # token UQ head
        self.token_uq_dim = token_uq_dim
        if token_uq_dim > 0:
            self.token_uq_head = UncertaintyHead(in_dim=hidden_dim, hidden=hidden, out_dim=token_uq_dim, dropout=dropout)
        else:
            self.token_uq_head = None

        # global direct MLP
        self.global_direct_dim = global_direct_dim
        if global_direct_dim > 0:
            self.global_direct_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, global_direct_dim)
            )
        else:
            self.global_direct_mlp = None

        # global UQ head
        self.global_uq_dim = global_uq_dim
        if global_uq_dim > 0:
            self.global_uq_head = UncertaintyHead(in_dim=hidden_dim, hidden=hidden, out_dim=global_uq_dim, dropout=dropout)
        else:
            self.global_uq_head = None

    def forward(self, local_feats):
        """Forward.

        Args:
          local_feats: [B, N, H]
        Returns:
          token_direct, global_direct, token_uq, global_uq
        """
        B, N, H = local_feats.shape

        token_direct = None
        token_uq = None
        global_direct = None
        global_uq = None

        if self.token_direct_mlp is not None:
            token_direct = self.token_direct_mlp(local_feats)  # [B,N,D]

        if self.token_uq_head is not None:
            # Uncertainty head applied per token
            tok_mean, tok_logvar = self.token_uq_head(local_feats)
            token_uq = (tok_mean, tok_logvar)

        # compute global pooled feature
        if self.pool == 'mean':
            global_in = local_feats.mean(dim=1)
        elif self.pool == 'max':
            global_in = local_feats.max(dim=1)[0]
        else:
            raise ValueError('unsupported pool')

        if self.global_direct_mlp is not None:
            global_direct = self.global_direct_mlp(global_in)

        if self.global_uq_head is not None:
            g_mean, g_logvar = self.global_uq_head(global_in)
            global_uq = (g_mean, g_logvar)

        return token_direct, global_direct, token_uq, global_uq
