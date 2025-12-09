import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

# optional: need for building radius graph when edge_index is missing
try:
    from torch_geometric.nn import radius_graph
except Exception:
    radius_graph = None
    # if radius_graph is unavailable, user must provide edges


class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim*2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_feat):
        # edge_index: [2, E], edge_feat: [E, F_e]
        row, col = edge_index  # i <- j
        m_input = torch.cat([x[row], x[col], edge_feat], dim=-1)
        m_ij = self.edge_mlp(m_input)
        agg = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))
        x = self.node_mlp(torch.cat([x, agg], dim=-1))
        return x


class SE(nn.Module):
    """
    GNN + 真正枚举三体角的径向+角向 SE(3) 编码器（支持 batch）
    forward 输入:
        x: [N, node_dim]
        pos: [N, 3]
        edge_index: [2, E] (or None)
        edge_attr: [E, edge_dim] or None
        triplets: optional (not used here; we enumerate from edge_index)
        batch: [N] node->graph index (optional; if None treats as single graph)
    输出 dict:
        local_se: [N, hidden_dim]
        global_se: [num_graphs, hidden_dim]
        pos: [N, 3]
    """
    def __init__(self, node_dim, edge_dim, hidden_dim,
                 num_layers=3, num_radial=16, num_angular=8, cutoff=8.0):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        # edge_proj expects (radial + angular + edge_attr_dim)
        self.edge_proj = nn.Linear((num_radial + num_angular) + (edge_dim if edge_dim is not None else 0),
                                   hidden_dim)
        self.layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.global_readout = nn.Linear(hidden_dim, hidden_dim)
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.cutoff = cutoff

    def radial_encoding(self, dist):
        """
        dist: [E]
        return [E, num_radial]
        """
        centers = torch.linspace(0.0, self.cutoff, self.num_radial, device=dist.device)
        widths = (centers[1] - centers[0]) + 1e-12
        return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (widths ** 2))

    def angular_encoding(self, pos, edge_index):
        """
        For each edge e: (i <- j), compute angles ∠ijk for all k in N(j)\\{i},
        convert cos(theta) to RBFs and average over k.
        Returns: [E, num_angular]
        """
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((0, self.num_angular), device=pos.device)

        row, col = edge_index  # both shape [E]
        E = row.size(0)
        N = pos.size(0)

        # build neighbor lists for each node j: neighbors[j] = list of nodes n s.t. there is edge (n <- j)
        neighbors = [[] for _ in range(N)]
        # we iterate edges once
        for e in range(E):
            i = int(row[e].item()); j = int(col[e].item())
            neighbors[j].append(i)

        angles = []
        centers = torch.linspace(-1.0, 1.0, self.num_angular, device=pos.device)
        widths = (centers[1] - centers[0]) + 1e-12

        for e in range(E):
            i = int(row[e].item()); j = int(col[e].item())
            neigh_k = [k for k in neighbors[j] if k != i]
            if len(neigh_k) == 0:
                # no valid k -> zero vector
                angles.append(torch.zeros(self.num_angular, device=pos.device))
                continue

            # v1 = pos[i] - pos[j], shape [3]
            v1 = pos[i] - pos[j]  # [3]
            # v2s = pos[k] - pos[j], shape [num_k,3]
            k_idx = torch.tensor(neigh_k, dtype=torch.long, device=pos.device)
            v2s = pos[k_idx] - pos[j].unsqueeze(0)  # [num_k,3]

            # normalize
            v1_norm = v1 / (v1.norm() + 1e-8)
            v2s_norm = v2s / (v2s.norm(dim=-1, keepdim=True) + 1e-8)  # [num_k,3]

            cos_theta = (v1_norm.unsqueeze(0) * v2s_norm).sum(dim=-1)  # [num_k], values in [-1,1]
            # RBF over cos_theta
            ang_k = torch.exp(-((cos_theta.unsqueeze(-1) - centers) ** 2) / (widths ** 2))  # [num_k, num_angular]
            ang_feat = ang_k.mean(dim=0)  # aggregate over k -> [num_angular]
            angles.append(ang_feat)

        angular_feat = torch.stack(angles, dim=0)  # [E, num_angular]
        return angular_feat

    def ensure_edge_index(self, edge_index, pos, batch):
        """
        If edge_index is None or empty and radius_graph is available, build radius graph using cutoff.
        Returns edge_index [2, E].
        """
        if edge_index is None or edge_index.numel() == 0:
            if radius_graph is None:
                raise RuntimeError("edge_index is empty and torch_geometric.nn.radius_graph not available. Provide edges.")
            # build radius graph (requires batch vector if multiple graphs)
            if batch is None:
                # treat single graph
                edge_index = radius_graph(pos, r=self.cutoff, loop=False)
            else:
                edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
        return edge_index

    def forward(self, x, pos, edge_index=None, edge_attr=None, triplets=None, batch=None):
        """
        x: [N, node_dim]
        pos: [N, 3]
        edge_index: [2, E] or None
        edge_attr: [E, edge_dim] or None
        batch: [N] long tensor (optional) for graph batches
        """
        # basic sanity checks to produce clearer diagnostics on shape mismatches
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"SE.forward expected tensor 'x', got {type(x)}")
        if x.dim() != 2:
            raise RuntimeError(f"SE.forward expected 'x' to be 2D [N, node_dim], got shape {tuple(x.shape)}")
        # ensure node_proj input dim matches
        try:
            expected_node_in = self.node_proj.in_features
        except Exception:
            expected_node_in = None
        if expected_node_in is not None and x.size(1) != expected_node_in:
            raise RuntimeError(f"SE.node_proj input dim mismatch: expected {expected_node_in}, got {x.size(1)}")

        # project nodes
        x = self.node_proj(x)  # [N, hidden_dim]

        # ensure edges exist (if user wants automatic radius-based edges)
        edge_index = self.ensure_edge_index(edge_index, pos, batch)  # may raise if not possible
        row, col = edge_index
        E = row.size(0)

        # radial features per edge
        rel_pos = pos[row] - pos[col]  # [E,3]
        dist = torch.norm(rel_pos, dim=-1)  # [E]
        radial_feat = self.radial_encoding(dist)  # [E, num_radial]

        # angular features per edge (exact triad enumeration)
        angular_feat = self.angular_encoding(pos, edge_index)  # [E, num_angular]

        # edge_attr handling
        if edge_attr is None:
            edge_attr_tensor = None
        else:
            # ensure shape [E, D]
            if edge_attr.dim() == 1:
                edge_attr_tensor = edge_attr.unsqueeze(-1)
            else:
                edge_attr_tensor = edge_attr

        # build full edge feature [E, num_radial + num_angular + edge_dim]
        if edge_attr_tensor is not None:
            edge_feat_raw = torch.cat([radial_feat, angular_feat, edge_attr_tensor], dim=-1)
        else:
            edge_feat_raw = torch.cat([radial_feat, angular_feat], dim=-1)

        # sanity check: ensure the projected linear expects the same in_features
        try:
            expected_edge_in = self.edge_proj.in_features
        except Exception:
            expected_edge_in = None
        if expected_edge_in is not None and edge_feat_raw.size(1) != expected_edge_in:
            # If the expected in-features is absurdly large (likely due to prior bad initialization
            # or accidental overwrite), attempt a conservative auto-fix so training can continue.
            try:
                big_threshold = 100000
                if isinstance(expected_edge_in, int) and expected_edge_in >= big_threshold:
                    old_ep = self.edge_proj
                    out_f = getattr(old_ep, 'out_features', None) or self.edge_proj.weight.size(0)
                    # re-create a compatible Linear on the same device/dtype
                    device = edge_feat_raw.device
                    new_ep = nn.Linear(edge_feat_raw.size(1), out_f).to(device)
                    self.edge_proj = new_ep
                    # log warning so user can inspect model initialization path
                    try:
                        import logging
                        logging.warning(f"[SE] auto-reinitialized edge_proj due to unexpected in_features={expected_edge_in}; new in_features={edge_feat_raw.size(1)} out_features={out_f}")
                    except Exception:
                        print(f"[SE] auto-reinitialized edge_proj due to unexpected in_features={expected_edge_in}; new in_features={edge_feat_raw.size(1)} out_features={out_f}")
                else:
                    raise RuntimeError(
                        f"SE.edge_proj input dim mismatch: expected {expected_edge_in}, got {edge_feat_raw.size(1)}; "
                        f"radial:{radial_feat.shape}, angular:{angular_feat.shape}, edge_attr:{None if edge_attr_tensor is None else edge_attr_tensor.shape}, edge_index:{None if edge_index is None else edge_index.shape}, pos:{pos.shape}"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                # If auto-fix failed, raise original informative error
                raise RuntimeError(
                    f"SE.edge_proj input dim mismatch and auto-fix failed ({e}): expected {expected_edge_in}, got {edge_feat_raw.size(1)}"
                )

        # linear project edge features to hidden_dim
        edge_feat = self.edge_proj(edge_feat_raw)  # [E, hidden_dim]

        # GNN layers (message passing)
        for layer in self.layers:
            x = layer(x, edge_index, edge_feat)

        # outputs
        local_se = x  # [N, hidden_dim]
        if batch is None:
            graph_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            graph_idx = batch

        global_se = scatter_add(x, graph_idx, dim=0)  # [num_graphs, hidden_dim]
        global_se = self.global_readout(global_se)

        return {
            "local_se": local_se,
            "global_se": global_se,
            "batch": batch,
            "pos": pos,
        }


class Fusion(nn.Module):
    """
    QM/MD 多粒度融合模块
    输入（来自各自的 SE 编码器）：
        qm_out: dict with keys {"local_se": [Nq, H], "global_se": [B, H], "pos": [Nq,3]}
        md_out: dict with keys {"local_se": [Nm, H], "global_se": [B, H], "pos": [Nm,3]}
    输出：
        {
            "local_fusion": [local_target_len, fusion_hidden_dim] (or [B, local_target_len, fusion_hidden_dim] if batched extension),
            "global_fusion": [B, fusion_hidden_dim],
            "weights": {"qm": [B,1], "md": [B,1]}
        }
    Notes:
    - local_target_len: 把局部节点数对齐到固定长度（默认128）。
    - 如果 global_se 的 batch 大小不是 1，请确保 QM/MD batch 对齐（同一批次的图顺序一致）。
    """
    def __init__(self, hidden_dim, fusion_hidden_dim=128, local_target_len=128, use_confidence=True,
                 preserve_global_scale: bool = False, residual_global_scale: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.local_target_len = local_target_len
        self.use_confidence = use_confidence

        # 局部融合 MLP（接受拼接后的 2*hidden_dim）
        self.local_fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.SiLU()
        )

        # 全局融合 MLP（接受拼接后的 2*hidden_dim）
        self.global_fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.SiLU()
        )

        # 置信度推断器（从 global_se -> scalar confidence）
        if self.use_confidence:
            self.confidence_mean_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.confidence_logvar_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        # optional controls
        self.preserve_global_scale = bool(preserve_global_scale)
        self.residual_global_scale = float(residual_global_scale)
        if self.residual_global_scale and self.residual_global_scale > 0.0:
            self.residual_proj = nn.Linear(hidden_dim, fusion_hidden_dim)

        # 更稳健的归一化：LayerNorm 保留可学习的缩放和平移参数
        self.global_layernorm = nn.LayerNorm(fusion_hidden_dim)

    def align_local_features(self, x: torch.Tensor):
        """
        把节点数对齐到 local_target_len
        x: [N, H]
        返回: [local_target_len, H]
        使用 adaptive_avg_pool1d 在长度维度上做池化
        """
        if x.size(0) == 0:
            return torch.zeros((self.local_target_len, self.hidden_dim), device=x.device, dtype=x.dtype)
        xp = x.T.unsqueeze(0)  # [1, H, N]
        pooled = F.adaptive_avg_pool1d(xp, self.local_target_len).squeeze(0)  # [H, local_target_len]
        return pooled.T  # [local_target_len, H]

    def align_local_features_batch(self, x: torch.Tensor, batch: torch.Tensor):
        """
        按图批次对每个图分别池化到固定长度。
        x: [N, H], batch: [N] (值从 0..B-1)
        返回: [B, L, H]
        """
        if batch is None:
            out = self.align_local_features(x)
            return out.unsqueeze(0)  # [1, L, H]

        device = x.device
        B = int(batch.max().item()) + 1
        H = x.size(1)
        L = self.local_target_len
        out = x.new_zeros((B, L, H))

        for b in range(B):
            mask = (batch == b)
            xb = x[mask]
            if xb.size(0) == 0:
                continue
            xp = xb.T.unsqueeze(0)  # [1, H, Ni]
            pooled = F.adaptive_avg_pool1d(xp, L).squeeze(0)  # [H, L]
            out[b] = pooled.T

        return out

    def compute_confidences(self, global_qm, global_md):
        """
        稳定版置信度与自适应权重计算，防止方差塌陷/数值下溢
        """
        if global_qm.dim() == 1:
            global_qm = global_qm.unsqueeze(0)
        if global_md.dim() == 1:
            global_md = global_md.unsqueeze(0)

        # --- 1. 对输入embedding进行标准化 ---
        global_qm = (global_qm - global_qm.mean(dim=1, keepdim=True)) / (global_qm.std(dim=1, keepdim=True) + 1e-6)
        global_md = (global_md - global_md.mean(dim=1, keepdim=True)) / (global_md.std(dim=1, keepdim=True) + 1e-6)

        # --- 2. 计算均值与logvar ---
        qm_mu = self.confidence_mean_mlp(global_qm)
        md_mu = self.confidence_mean_mlp(global_md)
        qm_logvar = self.confidence_logvar_mlp(global_qm)
        md_logvar = self.confidence_logvar_mlp(global_md)

        # --- 3. 限制logvar的范围 ---
        qm_logvar = torch.clamp(qm_logvar, min=-5.0, max=5.0)
        md_logvar = torch.clamp(md_logvar, min=-5.0, max=5.0)

        # --- 4. 计算方差（确保不会为0）---
        qm_var = torch.exp(qm_logvar).clamp(min=1e-4, max=1e2)
        md_var = torch.exp(md_logvar).clamp(min=1e-4, max=1e2)

        # --- 5. 使用精度(precision=1/var)作为权重 ---
        qm_prec = (1.0 / qm_var).clamp(min=1e-8, max=1e8)
        md_prec = (1.0 / md_var).clamp(min=1e-8, max=1e8)
        total_prec = (qm_prec + md_prec)
        qm_w = qm_prec / total_prec
        md_w = md_prec / total_prec

        # --- 6. 关键调试输出（改为显示精度与权重） ---
        try:
            print(f"[Fusion] mean_qm={qm_mu.mean():.4f}, prec_qm={qm_prec.mean():.4f}, mean_md={md_mu.mean():.4f}, prec_md={md_prec.mean():.4f}, weights=({qm_w.mean():.4f}, {md_w.mean():.4f})")
        except Exception:
            pass

        return qm_w, md_w

    def forward(self, qm_out: dict, md_out: dict):
        """
        qm_out and md_out are dicts from SE encoder
        """
        # ---- local alignment ----
        qm_local = qm_out["local_se"]  # [Nq, H]
        md_local = md_out["local_se"]  # [Nm, H]

        # 支持批次信息：若提供 batch，则为每个图分别池化
        qm_batch = qm_out.get('batch', None) if isinstance(qm_out, dict) else None
        md_batch = md_out.get('batch', None) if isinstance(md_out, dict) else None

        if qm_batch is not None:
            qm_local_aligned = self.align_local_features_batch(qm_local, qm_batch)  # [B, L, H]
        else:
            qm_local_aligned = self.align_local_features(qm_local).unsqueeze(0)  # [1, L, H]

        if md_batch is not None:
            md_local_aligned = self.align_local_features_batch(md_local, md_batch)  # [B, L, H]
        else:
            md_local_aligned = self.align_local_features(md_local).unsqueeze(0)  # [1, L, H]

        # concat along feature dim -> [B, L, 2H]
        # 如果 batch 大小不一致则尝试广播第一个维度
        if qm_local_aligned.size(0) != md_local_aligned.size(0):
            if qm_local_aligned.size(0) == 1:
                qm_local_aligned = qm_local_aligned.repeat(md_local_aligned.size(0), 1, 1)
            elif md_local_aligned.size(0) == 1:
                md_local_aligned = md_local_aligned.repeat(qm_local_aligned.size(0), 1, 1)
            else:
                raise ValueError(f"local aligned batch mismatch: qm {qm_local_aligned.size(0)} vs md {md_local_aligned.size(0)}")

        local_pair = torch.cat([qm_local_aligned, md_local_aligned], dim=-1)  # [B, L, 2H]
        # local_fusion_mlp 会在最后一维上应用线性层 -> 输出 [B, L, fusion_hidden_dim]
        local_fusion = self.local_fusion_mlp(local_pair)

        # ---- global fusion ----
        g_qm = qm_out["global_se"]  # [B, H] or [H]
        g_md = md_out["global_se"]  # [B, H] or [H]

        # If global tensors are single-graph vectors [H], unsqueeze to [1,H]
        if g_qm.dim() == 1:
            g_qm = g_qm.unsqueeze(0)
        if g_md.dim() == 1:
            g_md = g_md.unsqueeze(0)

        # check batch size compatibility
        if g_qm.size(0) != g_md.size(0):
            # 若 batch 大小不一致，尝试广播：若其中一个是 1，则 repeat
            if g_qm.size(0) == 1:
                g_qm = g_qm.repeat(g_md.size(0), 1)
            elif g_md.size(0) == 1:
                g_md = g_md.repeat(g_qm.size(0), 1)
            else:
                raise ValueError(f"global_se batch mismatch: qm {g_qm.size(0)} vs md {g_md.size(0)}")

        # compute confidences (optional)
        if self.use_confidence:
            qm_w, md_w = self.compute_confidences(g_qm, g_md)  # each [B,1]
        else:
            # uniform weights
            qm_w = torch.full((g_qm.size(0), 1), 0.5, device=g_qm.device)
            md_w = torch.full((g_qm.size(0), 1), 0.5, device=g_qm.device)

        # weighted concatenation of global features
        # apply scalar weights to vectors (broadcast)
        g_qm_w = g_qm * qm_w  # [B,H]
        g_md_w = g_md * md_w  # [B,H]
        global_pair = torch.cat([g_qm_w, g_md_w], dim=-1)  # [B, 2H]
        global_fusion = self.global_fusion_mlp(global_pair)  # [B, fusion_hidden_dim]

        # --- residual connection to preserve QM/MD original variance ---
        avg_raw = (g_qm + g_md) / 2.0  # [B, hidden_dim]
        # Ensure avg_raw and global_fusion have same feature dim: project avg_raw if necessary
        if self.hidden_dim != self.fusion_hidden_dim:
            if not hasattr(self, 'residual_proj_raw'):
                # lazy create projection layer to map raw global -> fusion dim
                # ensure the new layer is created on the same device as the input tensor
                device = avg_raw.device if isinstance(avg_raw, torch.Tensor) else None
                if device is not None:
                    self.residual_proj_raw = nn.Linear(self.hidden_dim, self.fusion_hidden_dim).to(device)
                else:
                    self.residual_proj_raw = nn.Linear(self.hidden_dim, self.fusion_hidden_dim)
            avg_raw_proj = self.residual_proj_raw(avg_raw)
        else:
            avg_raw_proj = avg_raw

        global_fusion = 0.5 * avg_raw_proj + 0.5 * global_fusion

        # Use LayerNorm (learnable) instead of manual per-sample mean/std normalization
        try:
            global_fusion = self.global_layernorm(global_fusion)
        except Exception:
            # fallback: leave as-is
            pass

        # --- optional residual from raw globals to preserve magnitude ---
        if getattr(self, 'residual_global_scale', 0.0) and self.residual_global_scale > 0.0:
            # residual input: average of qm/md raw global embeddings
            avg_raw = (g_qm + g_md) / 2.0
            # project
            try:
                res_proj = self.residual_proj(avg_raw)
                global_fusion = global_fusion + self.residual_global_scale * res_proj
            except Exception:
                pass

        # --- optional: preserve global scale (std) to avoid collapse ---
        scale_used = 1.0
        if getattr(self, 'preserve_global_scale', False):
            try:
                # compute per-batch average std of inputs and output
                in_std_qm_tensor = g_qm.std(dim=1).mean()
                in_std_md_tensor = g_md.std(dim=1).mean()
                in_std = float(((in_std_qm_tensor + in_std_md_tensor) / 2.0).detach().cpu().item())
                out_std_tensor = global_fusion.std(dim=1).mean()
                out_std = float(out_std_tensor.detach().cpu().item())
                if out_std > 1e-8:
                    scale = float(in_std / (out_std + 1e-8))
                    # clamp to reasonable range
                    scale = max(min(scale, 10.0), 0.1)
                    global_fusion = global_fusion * scale
                    scale_used = float(scale)
                # pass out_std back as tensor for fusion_stats
                out_std = out_std
                in_std_qm = float(in_std_qm_tensor.detach().cpu().item())
                in_std_md = float(in_std_md_tensor.detach().cpu().item())
            except Exception as e:
                print(f"[Fusion::preserve_global_scale] exception: {e}")
                in_std_qm = None
                in_std_md = None
                out_std = None


        def _safe_float(v):
            try:
                if isinstance(v, torch.Tensor):
                    return float(v.detach().cpu().item())
                return float(v)
            except Exception:
                return None

        fusion_stats = {
            'in_std_qm': _safe_float(in_std_qm) if 'in_std_qm' in locals() else None,
            'in_std_md': _safe_float(in_std_md) if 'in_std_md' in locals() else None,
            'out_std': _safe_float(out_std) if 'out_std' in locals() else None,
            'scale_used': float(scale_used)
        }

        return {
            "local_fusion": local_fusion,      # [B, L, fusion_hidden_dim]
            "global_fusion": global_fusion    # [B, fusion_hidden_dim]
            ,
            'fusion_stats': fusion_stats
        }


