import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# 基础注意力模块
# =====================
class UncertaintyAttention(nn.Module):
    """基础注意力模块，用于 QM→MD 或 MD→QM 单方向注意力
    增加不确定性权重处理模式（mult（默认）：乘法；logadd：对数空间加权）
    """
    def __init__(self, hidden_dim=128, dropout=0.1, uncert_mode: str = 'mult', uncert_scale: float = 1.0, uncert_eps: float = 1e-6, temp: float = 1.0):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 可学习的不确定性敏感度（类似论文中的 τ）
        self.tau = nn.Parameter(torch.tensor(1.0))
        # uncert handling mode
        self.uncert_mode = str(uncert_mode)
        self.uncert_scale = float(uncert_scale)
        self.uncert_eps = float(uncert_eps)
        # temperature for attention scaling
        self.temp = float(temp)
        # self.linear1 = nn.Linear(6534, 10304)
    def forward(self, query_feat, key_feat, qm_uncert=None, md_uncert=None, uncert_weight=None, mask=None):
        """
        Args:
            query_feat: [B, Nq, H]
            key_feat: [B, Nk, H]
            qm_uncert: (optional) [B, Nq] QM 局部不确定性向量
            md_uncert: (optional) [B, Nk] MD 局部不确定性向量
            uncert_weight: (optional) [B, Nq, Nk] 预计算的不确定性权重（优先级高于 qm/md_uncert）
            mask: (optional) 用于屏蔽 attention 的 mask。支持多种形状（见代码注释）

        Returns:
            out: [B, Nq, H]
            attn_weights: [B, Nq, Nk]
        """
        B, Nq, H = query_feat.shape
        _, Nk, _ = key_feat.shape

        # === Step 1. 线性投影 ===
        Q = self.q_proj(query_feat)  # [B, Nq, H]
        K = self.k_proj(key_feat)    # [B, Nk, H]
        V = self.v_proj(key_feat)    # [B, Nk, H]

        # === Step 2. 注意力 logits ===
        # To avoid OOM we compute attention in query-chunks when B*Nq*Nk is large.
        num_elems = int(B) * int(Nq) * int(Nk)
        CHUNK_ELEM_THRESHOLD = 10_000_000  # ~10M elements (~40MB float32)
        use_chunked = num_elems > CHUNK_ELEM_THRESHOLD

        if not use_chunked:
            attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (H ** 0.5)  # [B, Nq, Nk]

            # === Step 3. 计算/合并不确定性权重 ===
            if uncert_weight is None:
                if (qm_uncert is not None) and (md_uncert is not None):
                    combined = qm_uncert.unsqueeze(-1) + md_uncert.unsqueeze(1)
                    uncert_weight = torch.sigmoid(self.tau * combined)
                else:
                    uncert_weight = torch.ones_like(attn_logits)
        else:
            # Chunked path: compute logits/uncert/mask/softmax per query-chunk to keep memory low
            # Determine chunk size so that B * chunk * Nk <= CHUNK_ELEM_THRESHOLD
            chunk = max(1, CHUNK_ELEM_THRESHOLD // max(1, (B * Nk)))
            attn_weights_chunks = []
            attended_chunks = []

            q_start = 0
            while q_start < Nq:
                q_end = min(Nq, q_start + chunk)
                Q_chunk = Q[:, q_start:q_end, :]  # [B, qchunk, H]

                # logits for this chunk
                logits_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) / (H ** 0.5)  # [B, qchunk, Nk]

                # compute uncert weight for this chunk (if available)
                if uncert_weight is None:
                    if (qm_uncert is not None) and (md_uncert is not None):
                        qm_c = qm_uncert[:, q_start:q_end].unsqueeze(-1)  # [B, qchunk, 1]
                        combined_chunk = qm_c + md_uncert.unsqueeze(1)  # [B, qchunk, Nk]
                        uncert_chunk = torch.sigmoid(self.tau * combined_chunk)
                    else:
                        uncert_chunk = None
                else:
                    # slice provided uncert_weight per-chunk (fallback to pad/truncate handled below)
                    try:
                        uncert_chunk = uncert_weight[:, q_start:q_end, :]
                    except Exception:
                        # if slicing fails, create a chunk of ones
                        uncert_chunk = None

                # apply uncert modulation if present
                if uncert_chunk is not None:
                    # try to ensure shape matches logits_chunk
                    if uncert_chunk.shape != logits_chunk.shape:
                        # pad/truncate to target shape conservatively
                        new = torch.ones_like(logits_chunk)
                        sb = min(uncert_chunk.size(0), new.size(0))
                        sq = min(uncert_chunk.size(1), new.size(1))
                        sk = min(uncert_chunk.size(2), new.size(2))
                        new[:sb, :sq, :sk] = uncert_chunk[:sb, :sq, :sk]
                        uncert_chunk = new
                    # sanitize uncert_chunk to avoid NaN/inf and avoid zero rows
                    try:
                        uncert_chunk = torch.nan_to_num(uncert_chunk, nan=1.0, posinf=1e6, neginf=1e-6)
                        try:
                            uncert_chunk = uncert_chunk.clamp(min=1e-6, max=1e6)
                        except Exception:
                            uncert_chunk = torch.clamp(uncert_chunk, min=1e-6, max=1e6)
                        # ensure no all-zero rows (which would force uniform softmax)
                        if uncert_chunk.dim() == 3:
                            row_sum = uncert_chunk.abs().sum(dim=-1, keepdim=True)
                            zero_row = (row_sum <= 1e-12)
                            if zero_row.any():
                                uncert_chunk = torch.where(zero_row.expand_as(uncert_chunk), torch.ones_like(uncert_chunk) * 1e-6, uncert_chunk)
                    except Exception:
                        pass
                    if hasattr(self, 'uncert_mode') and self.uncert_mode == 'logadd':
                        try:
                            logits_chunk = logits_chunk + (self.uncert_scale * torch.log(uncert_chunk + self.uncert_eps))
                        except Exception:
                            logits_chunk = logits_chunk * uncert_chunk
                    else:
                        logits_chunk = logits_chunk * uncert_chunk

                # temperature scaling for chunk
                logits_chunk = logits_chunk / self.temp

                # apply mask for this chunk if provided
                if mask is not None:
                    try:
                        if mask.dtype == torch.bool:
                            # Treat boolean mask as additive: False -> -inf (mask out), True -> keep
                            if mask.dim() == 2 and mask.shape[1] == Nk:
                                keep = mask.unsqueeze(1).expand(-1, q_end - q_start, -1)
                            elif mask.dim() == 2 and mask.shape[1] == (Nq if 'Nq' in locals() else Nq):
                                # [B, Nq] -> slice the current chunk and expand to Nk
                                keep = mask[:, q_start:q_end].unsqueeze(-1).expand(-1, q_end - q_start, Nk)
                            elif mask.dim() == 3:
                                keep = mask[:, q_start:q_end, :]
                            else:
                                # Fallback: try to broadcast; if fails, keep all
                                keep = torch.ones_like(logits_chunk, dtype=torch.bool)
                            logits_chunk = logits_chunk.masked_fill(~keep.to(logits_chunk.device), -1e9)
                        else:
                            # Non-bool mask: assume additive; try broadcasting first
                            if mask.shape == logits_chunk.shape:
                                logits_chunk = logits_chunk + mask
                            elif mask.dim() == 2 and mask.shape[1] == Nk:
                                logits_chunk = logits_chunk + mask.unsqueeze(1)
                            else:
                                # If incompatible, ignore gracefully
                                pass
                    except Exception:
                        pass

                # handle fully masked rows in chunk: use fallback (uniform attention)
                row_all_masked_chunk = (logits_chunk == -1e9).all(dim=-1, keepdim=True)
                if row_all_masked_chunk.any():
                    logits_chunk = torch.where(row_all_masked_chunk.expand_as(logits_chunk), torch.zeros_like(logits_chunk), logits_chunk)

                # softmax + dropout for chunk
                w_chunk = F.softmax(logits_chunk, dim=-1)
                w_chunk = self.dropout(w_chunk)

                # attended for chunk
                att_chunk = torch.matmul(w_chunk, V)  # [B, qchunk, H]

                attn_weights_chunks.append(w_chunk)
                attended_chunks.append(att_chunk)

                q_start = q_end

            # concatenate results and return (we've handled weights, mask, softmax)
            attn_weights = torch.cat(attn_weights_chunks, dim=1)  # [B, Nq, Nk]
            attended = torch.cat(attended_chunks, dim=1)  # [B, Nq, H]
            out = self.out_proj(attended)
            return out, attn_weights
        

    # 确保形状匹配；如果形状不一致，尝试广播 / pad / truncate 以匹配 attn_logits 的 [B, Nq, Nk]
        if uncert_weight.shape != attn_logits.shape:
            # 尝试把常见的 [B, Nq] 或 [B, Nk] 扩展为 [B, Nq, Nk]
            if uncert_weight.dim() == 2:
                # 如果是 [B, Nq] -> 当作 qm_uncert
                if uncert_weight.size(0) == B and uncert_weight.size(1) == Nq:
                    uncert_weight = uncert_weight.unsqueeze(-1).expand(-1, -1, Nk)
                # 如果是 [B, Nk] -> 当作 md_uncert
                elif uncert_weight.size(0) == B and uncert_weight.size(1) == Nk:
                    uncert_weight = uncert_weight.unsqueeze(1).expand(-1, Nq, -1)
                else:
                    # 尝试 batch 为1 的广播
                    if uncert_weight.size(0) == 1 and uncert_weight.size(1) in (Nq, Nk):
                        if uncert_weight.size(1) == Nq:
                            uncert_weight = uncert_weight.unsqueeze(-1).expand(B, -1, Nk)
                        else:
                            uncert_weight = uncert_weight.unsqueeze(1).expand(B, Nq, -1)
                    else:
                        # fallback to full pad/truncate
                        pass

            if uncert_weight.dim() == 3 and uncert_weight.shape != attn_logits.shape:
                # 尝试处理 batch 尺寸不匹配（例如 batch=1 可重复）
                bw, aq, ak = uncert_weight.size()
                if bw == 1 and B > 1:
                    uncert_weight = uncert_weight.expand(B, aq, ak)

                # 最后一步：pad 或 truncate 到目标形状
                if uncert_weight.size(1) != Nq or uncert_weight.size(2) != Nk:
                    new = torch.zeros((B, Nq, Nk), device=attn_logits.device, dtype=uncert_weight.dtype)
                    src_b = min(uncert_weight.size(0), B)
                    src_q = min(uncert_weight.size(1), Nq)
                    src_k = min(uncert_weight.size(2), Nk)
                    new[:src_b, :src_q, :src_k] = uncert_weight[:src_b, :src_q, :src_k]
                    uncert_weight = new

            # 如果到这里仍然不匹配，抛出错误帮助定位
            if uncert_weight.shape != attn_logits.shape:
                raise ValueError(f"uncert_weight shape {uncert_weight.shape} incompatible with attention logits {attn_logits.shape}")

        # 将不确定性作为对 logits 的逐元素乘法调制（可认为是 α_ij <- α_ij * σ(...)）
        # sanitize uncert_weight to avoid inf/nan, large ranges or all-zero rows
        def _san_unc(u: torch.Tensor, eps: float = 1e-6, max_val: float = 1e3, row_normalize: bool = False):
            if u is None:
                return u
            # replace non-finite values
            u = torch.nan_to_num(u, nan=eps, posinf=max_val, neginf=eps)
            try:
                u = u.clamp(min=eps, max=max_val)
            except Exception:
                u = torch.clamp(u, min=eps, max=max_val)
            # avoid rows that are all zeros -> set to small eps
            if u.dim() == 3:
                row_sum = u.abs().sum(dim=-1, keepdim=True)
                zero_row = (row_sum <= eps * 1e-6)
                if zero_row.any():
                    u = torch.where(zero_row.expand_as(u), torch.ones_like(u) * eps, u)
                if row_normalize:
                    row_max = u.max(dim=-1, keepdim=True).values.clamp_min(eps)
                    u = u / row_max
            return u

        uncert_weight = _san_unc(uncert_weight, eps=1e-6, max_val=1e6, row_normalize=False)
        if hasattr(self, 'uncert_mode') and self.uncert_mode == 'logadd':
            try:
                attn_logits = attn_logits + (self.uncert_scale * torch.log(uncert_weight + self.uncert_eps))
            except Exception:
                attn_logits = attn_logits * uncert_weight
        else:
            attn_logits = attn_logits * uncert_weight

        # === Step 4. Temperature scaling ===
        attn_logits = attn_logits / self.temp

        # === Step 5. Mask 融合 ===
        # 支持的 mask 形式：
        # - additive mask（大负数，直接加到 logits），shape 可以是 [B, Nq, Nk] 或可广播到该形状
        # - boolean/float multiplicative mask 表示保留位置为1，屏蔽为0，常见 shape 为 [B, Nk] 或 [B, Nq, Nk]
        # 额外处理：如果用户不小心把被 mask 的特征（即 local_fusion * mask）作为 mask 传入，
        # 其形状通常为 [B, N, H]（最后一维为 hidden_dim）。在这种情况下我们恢复出 token 级的 bool mask
        if mask is not None:
            # 如果 mask 看起来像被 mask 的特征（float tensor，最后一维等于 hidden_dim），则恢复为 bool mask
            if isinstance(mask, torch.Tensor) and mask.dim() == 3 and mask.shape[2] == H and mask.dtype != torch.bool:
                # 被 mask 的特征通常是在被屏蔽的位置为0；我们用非零检测恢复 bool mask
                try:
                    restored_bool = (mask.abs().sum(dim=-1) > 0)
                    mask = restored_bool
                except Exception:
                    # 若恢复失败，则继续按原始 mask 处理（后续会抛出更明确的错误）
                    pass

        if mask is not None:
            # 如果是 bool mask，将其视为 additive mask：False -> -inf（屏蔽），True -> 保留
            if mask.dtype == torch.bool:
                # 构造与 attn_logits 同形状的 keep 掩码
                try:
                    if mask.dim() == 3 and mask.shape == attn_logits.shape:
                        keep = mask
                    elif mask.dim() == 2 and mask.shape[1] == Nk:
                        keep = mask.unsqueeze(1).expand(-1, Nq, -1)
                    elif mask.dim() == 2 and mask.shape[1] == Nq:
                        keep = mask.unsqueeze(-1).expand(-1, -1, Nk)
                    else:
                        # pad/truncate 到 [B, Nq, Nk]，默认保留（True）
                        keep = torch.ones((B, Nq, Nk), device=attn_logits.device, dtype=torch.bool)
                        if mask.dim() == 2:
                            sb = min(mask.size(0), B)
                            L = mask.size(1)
                            keep[:sb, :min(L, Nq), :].copy_(mask[:sb, :min(L, Nq)].unsqueeze(-1).expand(-1, -1, Nk))
                        elif mask.dim() == 3:
                            sb = min(mask.size(0), B)
                            sq = min(mask.size(1), Nq)
                            sk = min(mask.size(2), Nk)
                            keep[:sb, :sq, :sk].copy_(mask[:sb, :sq, :sk])
                except Exception:
                    keep = torch.ones_like(attn_logits, dtype=torch.bool)
                attn_logits = attn_logits.masked_fill(~keep, -1e9)

            else:
                # 非 bool 的 mask，可能是 additive（如来自 LocalMask 的 additive 格式），
                # 也可能是 multiplicative 的 float mask (0/1)。我们先尝试作为 additive：
                # additive mask: 先尝试直接广播相加
                try:
                    attn_logits = attn_logits + mask
                except Exception:
                    # 常见扩展
                    if mask.dim() == 2:
                        if mask.shape[1] == Nk:
                            try:
                                attn_logits = attn_logits + mask.unsqueeze(1)
                            except Exception:
                                pass
                        elif mask.shape[1] == Nq:
                            try:
                                attn_logits = attn_logits + mask.unsqueeze(-1)
                            except Exception:
                                pass
                    # 最后尝试 pad/truncate 填充 additive mask（用 0 填充）
                    if attn_logits.shape != mask.shape:
                        new_mask = torch.zeros((B, Nq, Nk), device=attn_logits.device, dtype=mask.dtype)
                        try:
                            sb = min(mask.size(0), B)
                            if mask.dim() == 2:
                                L = mask.size(1)
                                new_mask[:sb, :L, :L] = mask[:sb, :L].unsqueeze(-1).repeat(1, 1, min(L, Nk))
                                attn_logits = attn_logits + new_mask
                            elif mask.dim() == 3:
                                src_q = min(mask.size(1), Nq)
                                src_k = min(mask.size(2), Nk)
                                new_mask[:sb, :src_q, :src_k] = mask[:sb, :src_q, :src_k]
                                attn_logits = attn_logits + new_mask
                        except Exception:
                            raise ValueError(f"Cannot apply additive mask: attn_logits {attn_logits.shape}, mask {mask.shape}")

        # handle fully masked rows: use fallback (uniform attention)
        row_all_masked = (attn_logits == -1e9).all(dim=-1, keepdim=True)
        if row_all_masked.any():
            attn_logits = torch.where(row_all_masked.expand_as(attn_logits), torch.zeros_like(attn_logits), attn_logits)

        # === Step 5. Softmax ===
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, Nq, Nk]
        attn_weights = self.dropout(attn_weights)

        # === Step 6. 加权求和 ===
        attended = torch.matmul(attn_weights, V)  # [B, Nq, H]

        # === Step 7. 输出映射 ===
        out = self.out_proj(attended)  # [B, Nq, H]
        return out, attn_weights


# =====================
# 整合特征提取 + 双向注意力模块
# =====================
class UncertaintyCoAttention(nn.Module):
    """自动从 batch 中提取特征的双向不确定性加权注意力"""
    def __init__(self, fusion_hidden_dim=128, dropout=0.1, uncert_mode='mult', uncert_scale=1.0, uncert_eps=1e-6):
        super().__init__()
        self.fusion_hidden_dim = fusion_hidden_dim
        # pass uncert handling to attention
        self.uncert_mode = uncert_mode
        self.uncert_scale = float(uncert_scale)
        self.uncert_eps = float(uncert_eps)
        self.qm2md_attn = UncertaintyAttention(self.fusion_hidden_dim, dropout, uncert_mode=self.uncert_mode, uncert_scale=self.uncert_scale, uncert_eps=self.uncert_eps)
        self.md2qm_attn = UncertaintyAttention(self.fusion_hidden_dim, dropout, uncert_mode=self.uncert_mode, uncert_scale=self.uncert_scale, uncert_eps=self.uncert_eps)
        # projections to split fused features into QM/MD views
        self.local_qm_proj = nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim)
        self.local_md_proj = nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim)
        self.global_qm_proj = nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim)
        self.global_md_proj = nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim)

        # fusion projection heads for final outputs
        self.fusion_proj_qm = nn.Linear(2 * self.fusion_hidden_dim, self.fusion_hidden_dim)
        self.fusion_proj_md = nn.Linear(2 * self.fusion_hidden_dim, self.fusion_hidden_dim)
    # 新接口：直接接受 fusion 的 local/global 特征
    # local_fusion: [L, H] 或 [B, L, H]
    # global_fusion: [B, H]

    # =============== 前向传播 ===============
    def forward(self, local_fusion, global_fusion, qm_uncert=None, md_uncert=None, uncert_weight=None, mask=None):
        """
        新接口：接受 Fusion 输出的 local_fusion 与 global_fusion

        Args:
            local_fusion: [L, H] 或 [B, L, H]（若无 batch 维，则会对 global_fusion 的 batch 维重复）
            global_fusion: [B, H]
            qm_uncert: [B, L_qm] 或 None
            md_uncert: [B, L_md] 或 None
            uncert_weight: 预计算的权重矩阵（例如 [B, L_qm, L_md]）
            mask: 可选的 mask

        Returns: dict 包含融合后的 global/local 输出与注意力权重
        """
        # 规范化 local_fusion 到 [B, L, H]
        if local_fusion.dim() == 2:
            # [L, H] -> [1, L, H] -> repeat to B
            L, H = local_fusion.shape
            if global_fusion.dim() == 1:
                B = 1
            else:
                B = global_fusion.shape[0]
            local_f = local_fusion.unsqueeze(0).expand(B, -1, -1).contiguous()
        elif local_fusion.dim() == 3:
            local_f = local_fusion
            B, L, H = local_f.shape
        else:
            raise ValueError(f"Unsupported local_fusion shape: {local_fusion.shape}")

        # 规范化 global_fusion 到 [B, H]
        if global_fusion.dim() == 1:
            g_f = global_fusion.unsqueeze(0)
        else:
            g_f = global_fusion

        # 将 fused local 分为 QM/MD 两个视图（两个投影头）
        qm_local_view = self.local_qm_proj(local_f)  # [B, L, H]
        md_local_view = self.local_md_proj(local_f)  # [B, L, H]

        # Local-level attention: QM -> MD
        qm_local_fused, attn_local_qm2md = self.qm2md_attn(qm_local_view, md_local_view, qm_uncert, md_uncert, uncert_weight, mask)
        # MD -> QM
        # 如果提供了预计算 uncert_weight，需转置
        uncert_for_md2qm = None
        if uncert_weight is not None and isinstance(uncert_weight, torch.Tensor) and uncert_weight.dim() == 3:
            uncert_for_md2qm = uncert_weight.transpose(1, 2)
        else:
            uncert_for_md2qm = uncert_weight

        md_local_fused, attn_local_md2qm = self.md2qm_attn(md_local_view, qm_local_view, md_uncert, qm_uncert, uncert_for_md2qm, None if mask is None else (mask if mask.dim() != 3 else mask.transpose(1, 2)))

        # 融合 local 层输出（将原始 view 与注意力输出拼接并投影）
        qm_local_concat = torch.cat([qm_local_view, qm_local_fused], dim=-1)  # [B, L, 2H]
        md_local_concat = torch.cat([md_local_view, md_local_fused], dim=-1)
        qm_local_fused_out = self.fusion_proj_qm(qm_local_concat)  # [B, L, H]
        md_local_fused_out = self.fusion_proj_md(md_local_concat)  # [B, L, H]

        # ====== Global-level projections (用于真正的全局 gate) ======
        qm_global_view = self.global_qm_proj(g_f)  # [B, H]
        md_global_view = self.global_md_proj(g_f)  # [B, H]

        # Global-level attention: treat as sequence length 1，用于 gate
        qm_global_q = qm_global_view.unsqueeze(1)  # [B, 1, H]
        md_global_k = md_global_view.unsqueeze(1)  # [B, 1, H]

        qm_global_fused, attn_global_qm2md = self.qm2md_attn(
            qm_global_q, md_global_k,
            qm_uncert=None, md_uncert=None,
            uncert_weight=None, mask=None
        )  # attn_global_qm2md: [B,1,1]

        md_global_fused, attn_global_md2qm = self.md2qm_attn(
            md_global_k, qm_global_q,
            qm_uncert=None, md_uncert=None,
            uncert_weight=None, mask=None
        )  # attn_global_md2qm: [B,1,1]

        # ====== 专门用于可视化的 global token → local tokens map ======
        L = local_f.size(1)  # token 数

        # 全局 QM token → MD local tokens，得到 [B,1,L]
        qm_global_q_vis = qm_global_view.unsqueeze(1)  # [B,1,H]
        _, attn_global_qm2md_row = self.qm2md_attn(
            qm_global_q_vis, md_local_view,
            qm_uncert=None, md_uncert=None,
            uncert_weight=None, mask=mask
        )  # [B,1,L]

        # 全局 MD token → QM local tokens，得到 [B,1,L]
        md_global_q_vis = md_global_view.unsqueeze(1)  # [B,1,H]
        _, attn_global_md2qm_row = self.md2qm_attn(
            md_global_q_vis, qm_local_view,
            qm_uncert=None, md_uncert=None,
            uncert_weight=None,
            mask=None if mask is None else (mask if mask.dim() != 3 else mask.transpose(1, 2))
        )  # [B,1,L]

        # 将 1×L 的 row 扩展成 L×L，便于和 local 保持同一维度可视化
        attn_global_qm2md_map = attn_global_qm2md_row.repeat(1, L, 1)  # [B, L, L]
        attn_global_md2qm_map = attn_global_md2qm_row.repeat(1, L, 1)  # [B, L, L]

        # ====== Global 输出投影（用于后续预测） ======
        qm_global_concat = torch.cat([qm_global_view, qm_global_fused.squeeze(1)], dim=-1)  # [B, 2H]
        md_global_concat = torch.cat([md_global_view, md_global_fused.squeeze(1)], dim=-1)
        qm_global_fused_out = self.fusion_proj_qm(qm_global_concat)  # [B, H]
        md_global_fused_out = self.fusion_proj_md(md_global_concat)  # [B, H]

        return {
            "local": {
                "qm": qm_local_fused_out,
                "md": md_local_fused_out,
                "attn_qm2md": attn_local_qm2md,
                "attn_md2qm": attn_local_md2qm,
                "attn_map_qm2md": attn_local_qm2md,
                "attn_map_md2qm": attn_local_md2qm,
            },
            "global": {
                "qm": qm_global_fused_out,
                "md": md_global_fused_out,
                "attn_qm2md": attn_global_qm2md,       # 1×1 gate
                "attn_md2qm": attn_global_md2qm,       # 1×1 gate
                "attn_map_qm2md": attn_global_qm2md_map,  # ★ L×L，用于可视化
                "attn_map_md2qm": attn_global_md2qm_map,  # ★ L×L，用于可视化
            },
        }


