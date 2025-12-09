import torch
import torch.nn as nn

from DUW.utils.se3 import SE, Fusion
from DUW.utils.UE_QM import QMUncertaintyEstimator
from DUW.utils.UE_MD import MDUncertaintyEstimator
from DUW.utils.Mask import LocalMask, GlobalMask
from DUW.utils.UncertaintyCoAttention import UncertaintyCoAttention
from DUW.utils.gated_aggregator import GatedAggregator
from DUW.utils.prediction_head import PredictionHead
from DUW.utils.uncertaintyQ import UncertaintyQuantification


class duwnet(nn.Module):
    def __init__(self, batch_qm, batch_md, hidden_dim, num_layers, fusion_hidden_dim,
                 local_target_len, qmh5_file, mdh5_file, energy_threshold, mol_len, device,
                 mask_type='multiplicative', mask_value=-1e9, dropout=0.1, mode='dual',
                 debug: bool = False,
                 uncert_floor: float = 1e-6,
                 uncert_ceiling: float = 1e6,
                 uncert_row_normalize: bool = True,
                 uncert_smoothing_alpha: float = 0.05,
                 uncert_transform: str = 'none',
                 uncert_attn_mode: str = 'logadd',
                 uncert_attn_scale: float = 1.0,
                 uncert_attn_eps: float = 1e-6):
        """
        DUW网络主模块

        Args:
            batch_qm: QM数据批次
            batch_md: MD数据批次
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            fusion_hidden_dim: 融合层隐藏维度
            local_target_len: 局部特征目标长度
            qmh5_file: QM数据文件路径
            mdh5_file: MD数据文件路径
            energy_threshold: 能量阈值
            mask_type: mask类型，'multiplicative' 或 'additive'
            mask_value: additive mask的值（默认-1e9）
            mode: 模式选择，'dual'(默认), 'qm_only', 'md_only', 'early_fusion', 'late_fusion'
        """
        super(duwnet, self).__init__()
        self.mode = mode  # 添加模式属性
        # record device and move batch objects if possible
        self.device = device if device is not None else torch.device('cpu')

        # 只在初始化时临时使用 batch 来获取维度信息，不持久存储整个 batch
        # 这样可以避免显存累积
        if batch_qm is not None:
            try:
                batch_qm_temp = batch_qm.to(self.device) if hasattr(batch_qm, 'to') else batch_qm
            except Exception:
                batch_qm_temp = batch_qm
        else:
            batch_qm_temp = None

        if batch_md is not None:
            try:
                batch_md_temp = batch_md.to(self.device) if hasattr(batch_md, 'to') else batch_md
            except Exception:
                batch_md_temp = batch_md
        else:
            batch_md_temp = None

        # 初始化时设置，但会在训练时动态更新
        self.QMExtrator = batch_qm_temp
        self.MDSampler = batch_md_temp

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fusion_hidden_dim = fusion_hidden_dim
        self.local_target_len = local_target_len
        self.qmh5_file = qmh5_file
        self.mdh5_file = mdh5_file
        self.energy_threshold = energy_threshold
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.mol_len = mol_len
        self.dropout = dropout
        # 调试开关：开启后在 forward 中计算并打印/返回注意力与表示统计
        self.debug = bool(debug)
        # uncert sanitization params
        self.uncert_floor = float(uncert_floor)
        self.uncert_ceiling = float(uncert_ceiling)
        self.uncert_row_normalize = bool(uncert_row_normalize)
        self.uncert_smoothing_alpha = float(uncert_smoothing_alpha)
        self.uncert_transform = str(uncert_transform or 'none')
        # attention uncert handling
        self.uncert_attn_mode = str(uncert_attn_mode)
        self.uncert_attn_scale = float(uncert_attn_scale)
        self.uncert_attn_eps = float(uncert_attn_eps)

        # 初始化 SE 网络，处理 batch 为 None 的情况
        if batch_qm_temp is not None and hasattr(batch_qm_temp, 'x'):
            qm_node_dim = batch_qm_temp.x.shape[1]
            qm_edge_dim = (
                batch_qm_temp.edge_attr.size(-1)
                if hasattr(batch_qm_temp, "edge_attr") and batch_qm_temp.edge_attr is not None
                else 0
            )
        else:
            # 使用默认维度
            qm_node_dim = 9  # 默认 QM 特征维度
            qm_edge_dim = 0

        if batch_md_temp is not None and hasattr(batch_md_temp, 'x'):
            md_node_dim = batch_md_temp.x.shape[1]
            md_edge_dim = (
                batch_md_temp.edge_attr.size(-1)
                if hasattr(batch_md_temp, "edge_attr") and batch_md_temp.edge_attr is not None
                else 0
            )
        else:
            # 使用默认维度
            md_node_dim = 9  # 默认 MD 特征维度
            md_edge_dim = 0

        # 根据模式初始化不同的组件
        if self.mode in ['dual', 'qm_only', 'early_fusion', 'late_fusion']:
            self.se_qm = SE(
                node_dim=qm_node_dim,
                edge_dim=qm_edge_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
            # QM UE is initialized in the 'dual' block below; avoid duplicate init here.
        
        if self.mode in ['dual', 'md_only', 'early_fusion', 'late_fusion']:
            self.se_md = SE(
                node_dim=md_node_dim,
                edge_dim=md_edge_dim,
                hidden_dim=hidden_dim
            )
            # MD UE is initialized in the 'dual' block below; avoid duplicate init here.
        
        # Early Fusion 模式：使用简单拼接层替代复杂的 Fusion 和 Co-Attention
        if self.mode == 'early_fusion':
            self.early_fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Fusion 和 Mask 只在 dual 模式下需要
        if self.mode == 'dual':
            # Fusion: allow preserving global scale and residual add-in to avoid over-smoothing
            self.fusion = Fusion(hidden_dim=self.hidden_dim, fusion_hidden_dim=self.fusion_hidden_dim,
                                 local_target_len=self.local_target_len, preserve_global_scale=True, residual_global_scale=0.1)
            self.qm_UE = QMUncertaintyEstimator(self.qmh5_file)
            self.md_UE = MDUncertaintyEstimator(self.mdh5_file)

        # Mask, UncertaintyCoAttention 和 GatedAggregator 只在 dual 模式下需要
        if self.mode == 'dual':
            self.Localmask = LocalMask(
                qmh5_file=self.qmh5_file,
                hidden_dim=self.fusion_hidden_dim,
                local_target_len=self.local_target_len,
                energy_threshold=self.energy_threshold,
                mask_type=self.mask_type,
                mask_value=self.mask_value,
                device=self.device
            )
            self.Globalmask = GlobalMask(
                qmh5_file=self.qmh5_file,
                hidden_dim=self.fusion_hidden_dim,
                local_target_len=self.local_target_len,
                energy_threshold=self.energy_threshold,
                mask_type=self.mask_type,
                mask_value=self.mask_value,
                device=self.device
            )
            self.UncertaintyCoAttention = UncertaintyCoAttention(fusion_hidden_dim=self.fusion_hidden_dim,
                                                                 dropout=self.dropout,
                                                                 uncert_mode=self.uncert_attn_mode,
                                                                 uncert_scale=self.uncert_attn_scale,
                                                                 uncert_eps=self.uncert_attn_eps)
            self.gated_aggregator = GatedAggregator(fusion_hidden_dim=self.fusion_hidden_dim, predictor=True, preserve_fused_scale=True, fused_residual_scale=0.05)
            self.linea1 = nn.Linear(self.local_target_len, self.mol_len).to(self.device)

        # 初始化完成后，清理临时 batch 以释放显存
        # 这些会在训练时动态更新
        del batch_qm_temp, batch_md_temp
        # 不主动清空 GPU 缓存：遵循调用者的策略，避免在等待其他进程释放资源时干预 GPU 状态
        # prediction head: 根据模式调整输入维度
        if self.mode == 'dual':
            pred_head_hidden_dim = self.fusion_hidden_dim
        elif self.mode == 'early_fusion':
            pred_head_hidden_dim = self.hidden_dim  # early_fusion 输出也是 hidden_dim
        elif self.mode == 'late_fusion':
            # late_fusion 模式需要两个独立的 prediction head
            pred_head_hidden_dim = self.hidden_dim
        else:  # qm_only 或 md_only
            pred_head_hidden_dim = self.hidden_dim
        
        self.pred_head = PredictionHead(hidden_dim=pred_head_hidden_dim,
                                        token_direct_dim=0,
                                        token_uq_dim=0,
                                        global_direct_dim=1,
                                        global_uq_dim=1,
                                        hidden=128,
                                        dropout=self.dropout,
                                        pool='mean')
        
        # late_fusion 模式需要第二个独立的 prediction head（用于 MD）
        if self.mode == 'late_fusion':
            self.pred_head_md = PredictionHead(hidden_dim=pred_head_hidden_dim,
                                              token_direct_dim=0,
                                              token_uq_dim=0,
                                              global_direct_dim=1,
                                              global_uq_dim=1,
                                              hidden=128,
                                              dropout=self.dropout,
                                              pool='mean')
        
        # Uncertainty post-processor: normalize prediction/prediction_uq into canonical fields
        self.uq = UncertaintyQuantification(prefer_uq=True)
        # default merge behavior for PreHead + UQ -> final outputs
        # options can be: 'use_uq_when_available' (default), 'direct_only', 'uq_only', 'variance_weighted'
        self.output_merge_mode = 'variance_weighted'
        self.merge_eps = 1e-6

        # ensure all parameters and internal tensors are placed on the requested device
        if hasattr(self, "move_internal_tensors_to_device"):
            try:
                self.move_internal_tensors_to_device(self.device)
            except Exception:
                try:
                    self.to(self.device)
                except Exception:
                    pass
        else:
            try:
                self.to(self.device)
            except Exception:
                pass

    def merge_outputs(self, pred_raw: dict, uq_out: dict):
        """Merge PreHead direct outputs and UQ canonical outputs into a structured combined_output dict.

        Returns a dict with keys:
          - prediction_direct (optional)
          - prediction_mean (optional)
          - prediction_logvar (optional)
          - final_prediction
          - final_logvar (optional)
          - used_uq (bool)
          - raw_uq_out (original uq_out for debugging)

        Merge strategy is controlled by self.output_merge_mode.
        """
        combined_output = {}
        # canonical mean/logvar from uq_out if present
        pred_mean = None
        pred_logvar = None
        if isinstance(uq_out, dict) and "prediction_mean" in uq_out:
            pred_mean = uq_out.get("prediction_mean")
            pred_logvar = uq_out.get("prediction_logvar", None)

        # direct prediction from PreHead
        pred_direct = None
        if "prediction_direct" in pred_raw:
            pd = pred_raw["prediction_direct"]
            if isinstance(pd, torch.Tensor):
                pred_direct = pd.view(-1)
            else:
                try:
                    pred_direct = torch.tensor(pd)
                except Exception:
                    pred_direct = None

        # decide final prediction according to mode
        mode = self.output_merge_mode
        final_pred = None
        final_logvar = None
        used_uq = False
        # sanitize direct/mean before combining to avoid NaN propagation
        def _san(t: torch.Tensor):
            if t is None or not isinstance(t, torch.Tensor):
                return t
            if not torch.isfinite(t).all():
                # replace NaN/inf with zero; log fraction
                try:
                    nonfin = (~torch.isfinite(t)).float().mean().item()
                except Exception:
                    nonfin = None
                import logging
                logging.warning(f"[merge_outputs] replacing non-finite values in tensor (fraction={nonfin})")
                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return t
        pred_direct = _san(pred_direct)
        pred_mean = _san(pred_mean)
        pred_logvar = _san(pred_logvar)

        if mode == 'direct_only':
            final_pred = pred_direct if pred_direct is not None else pred_mean
            final_logvar = None if pred_direct is not None else pred_logvar
            used_uq = False if pred_direct is not None else (pred_mean is not None)
        elif mode == 'uq_only':
            final_pred = pred_mean
            final_logvar = pred_logvar
            used_uq = pred_mean is not None
        elif mode == 'variance_weighted':
            if pred_mean is not None and pred_logvar is not None and pred_direct is not None:
                var = torch.exp(torch.clamp(pred_logvar, min=-50.0, max=50.0))  # clamp to avoid overflow
                var = torch.nan_to_num(var, nan=1.0, posinf=1.0, neginf=1.0)
                alpha = 1.0 / (1.0 + var + self.merge_eps)
                alpha = alpha.view(-1)
                final_pred = alpha * pred_mean + (1.0 - alpha) * pred_direct.to(pred_mean.device)
                print("pred_mean:",pred_mean,"pred_direct",pred_direct,"final_pred",final_pred,"\n")
                final_logvar = pred_logvar
                used_uq = True
            else:
                final_pred = pred_mean if pred_mean is not None else pred_direct
                final_logvar = pred_logvar if pred_logvar is not None else None
                used_uq = pred_mean is not None
        else:  # default use_uq_when_available
            if pred_mean is not None:
                final_pred = pred_mean
                final_logvar = pred_logvar
                used_uq = True
            else:
                final_pred = pred_direct
                final_logvar = None
                used_uq = False

        # attach fields to combined_output
        if pred_direct is not None:
            combined_output['prediction_direct'] = pred_direct
        if pred_mean is not None:
            combined_output['prediction_mean'] = pred_mean
        if pred_logvar is not None:
            combined_output['prediction_logvar'] = pred_logvar
        combined_output['final_prediction'] = final_pred
        combined_output['final_logvar'] = final_logvar
        combined_output['used_uq'] = used_uq
        combined_output['raw_uq_out'] = uq_out
        return combined_output

    def _get_current_batch_ids(self):
        """Return a list of string IDs for the current batch (best-effort).

        This unifies the logic used in multiple places: prefer `ids`, then `id`,
        then `mol_id`, and finally `graph_idx` (supports tensor or scalar).
        Always returns a list of strings (empty list if nothing found).
        """
        try:
            # Prefer QMExtrator IDs when available (QM labels / metadata are canonical)
            qm = getattr(self, 'QMExtrator', None)
            if qm is not None:
                ids_attr = getattr(qm, 'ids', None) or getattr(qm, 'id', None) or getattr(qm, 'mol_id', None)
                if ids_attr is not None:
                    if isinstance(ids_attr, (list, tuple)):
                        return [str(x) for x in ids_attr]
                    if isinstance(ids_attr, torch.Tensor):
                        try:
                            return [str(x.item()) for x in ids_attr.view(-1)]
                        except Exception:
                            return [str(int(x)) for x in ids_attr.view(-1).tolist()]
                    return [str(ids_attr)]

            # Fallback to MDSampler if QMExtrator did not provide IDs
            ms = getattr(self, 'MDSampler', None)
            if ms is None:
                return []
            ids_attr = getattr(ms, 'ids', None) or getattr(ms, 'id', None) or getattr(ms, 'mol_id', None)

            # graph_idx fallback (index-style id) only if explicit
            if ids_attr is None and hasattr(ms, 'graph_idx'):
                idx_attr = getattr(ms, 'graph_idx')
                if isinstance(idx_attr, torch.Tensor):
                    return [str(int(x)) for x in idx_attr.view(-1).tolist()]
                else:
                    return [str(idx_attr)]

            if ids_attr is None:
                return []
            if isinstance(ids_attr, (list, tuple)):
                return [str(x) for x in ids_attr]
            if isinstance(ids_attr, torch.Tensor):
                try:
                    return [str(x.item()) for x in ids_attr.view(-1)]
                except Exception:
                    return [str(int(x)) for x in ids_attr.view(-1).tolist()]
            return [str(ids_attr)]
        except Exception:
            return []

    def forward(self, batch_md=None, batch_qm=None):
        """
        前向传播，支持五种模式：
        - 'dual': 使用 QM 和 MD 数据，完整的融合流程（Co-Attention）
        - 'qm_only': 只使用 QM 数据
        - 'md_only': 只使用 MD 数据
        - 'early_fusion': 简单拼接 QM 和 MD 特征
        - 'late_fusion': 分别预测 QM 和 MD，然后平均结果
        """
        # allow optional batch to be passed through forward to avoid relying on
        # external attribute setting. This makes the model DDP-friendly when
        # callers pass the batch as an argument (so DDP will move inputs and
        # moved_inputs won't be empty).
        if batch_md is not None:
            # 释放旧的 MDSampler 以避免显存累积
            if hasattr(self, 'MDSampler') and self.MDSampler is not None:
                del self.MDSampler
            try:
                # prefer to keep the batch on the configured device
                self.MDSampler = batch_md
            except Exception:
                pass

        if batch_qm is not None:
            # 释放旧的 QMExtrator 以避免显存累积
            if hasattr(self, 'QMExtrator') and self.QMExtrator is not None:
                del self.QMExtrator
            try:
                self.QMExtrator = batch_qm
            except Exception:
                pass

        # === QM-only 模式 ===
        if self.mode == 'qm_only':
            return self._forward_qm_only()
        
        # === MD-only 模式 ===
        elif self.mode == 'md_only':
            return self._forward_md_only()
        
        # === Early Fusion 模式 ===
        elif self.mode == 'early_fusion':
            return self._forward_early_fusion()
        
        # === Late Fusion 模式 ===
        elif self.mode == 'late_fusion':
            return self._forward_late_fusion()
        
        # === Dual 模式（原有逻辑）===
        else:
            return self._forward_dual()

    def _safe_se_md_call(self, **kwargs):
        """Safely call self.se_md with retry-on-error behaviour.

        If an exception occurs, this will clear caches, run GC, wait and retry
        until the call succeeds or a KeyboardInterrupt is raised.
        """
        import time
        import gc
        import logging

        attempt = 0
        while True:
            attempt += 1
            try:
                return self.se_md(**kwargs)
            except KeyboardInterrupt:
                # allow user to interrupt waiting
                raise
            except Exception as e:
                # log and attempt to free memory / wait before retrying
                logging.warning(f"[DUW::_safe_se_md_call] attempt {attempt} failed with error: {e}. Will wait and retry.")
                # Do NOT clear GPU cache here; other processes may be freeing memory.
                # We still run GC to release any CPU refs we can.
                try:
                    gc.collect()
                except Exception:
                    pass
                # exponential/backoff style wait (capped)
                wait_seconds = 10 if attempt < 6 else 30
                time.sleep(wait_seconds)
    
    def _forward_qm_only(self):
        """QM-only 模式的前向传播"""
        # 1. 编码QM数据
        outputs_qm = self.se_qm(
            x=self.QMExtrator.x,
            pos=self.QMExtrator.pos,
            edge_index=(self.QMExtrator.edge_index if hasattr(self.QMExtrator, "edge_index") else None),
            edge_attr=(self.QMExtrator.edge_attr if hasattr(self.QMExtrator, "edge_attr") else None),
            batch=(self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None)
        )
        
        # 2. 直接使用 outputs_qm 作为输入到 pred_head
        # outputs_qm 是字典，包含 'local_se' 和 'global_se'
        local_se = outputs_qm.get('local_se')  # [N, H]
        global_se = outputs_qm.get('global_se')  # [B, H]
        
        # 使用 global_se 作为 fused representation
        if global_se is not None:
            fused = global_se
        else:
            # 如果没有 global_se，从 local_se 聚合
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None
            if batch_idx is not None and local_se is not None:
                fused = global_mean_pool(local_se, batch_idx)
            elif local_se is not None:
                fused = local_se.mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("SE output contains neither local_se nor global_se")
        
        # Debug: 确保 fused 的 batch size 正确
        # fused 应该是 [B, H]，其中 B 是 batch 中的图数量
        if hasattr(self.QMExtrator, 'num_graphs'):
            expected_batch_size = self.QMExtrator.num_graphs
        elif hasattr(self.QMExtrator, 'batch'):
            expected_batch_size = self.QMExtrator.batch.max().item() + 1
        else:
            expected_batch_size = fused.size(0)
        
        # 如果 fused 的 batch size 不匹配，可能是因为某些图被过滤了
        # 我们需要确保输出的 batch size 与输入一致
        if fused.size(0) != expected_batch_size:
            print(f"Warning in _forward_qm_only: fused size {fused.size(0)} != expected {expected_batch_size}")
        
        # 3. 通过 prediction head
        token_direct, global_direct, token_uq, global_uq = self.pred_head(fused.unsqueeze(1))
        
        # 4. 构建输出
        pred_raw = {}
        if global_direct is not None:
            if global_direct.dim() > 1 and global_direct.size(-1) == 1:
                pred_raw["prediction_direct"] = global_direct.view(global_direct.shape[0])
            else:
                pred_raw["prediction_direct"] = global_direct
        if global_uq is not None:
            pred_raw["prediction_uq"] = global_uq
        
        # 5. UQ 后处理
        try:
            uq_out = self.uq(dict(pred_raw)) if pred_raw else {}
        except Exception as e:
            uq_out = dict(pred_raw)
            uq_out["uq_error"] = str(e)
        
        # 6. 合并输出
        combined_output = self.merge_outputs(pred_raw, uq_out)
        
        result = {
            "outputs_qm": outputs_qm,
            "PreHead": {
                "token_direct": token_direct,
                "global_direct": global_direct,
                "token_uq": token_uq,
                "global_uq": global_uq
            },
            "UQ_out": uq_out,
            "final_output": combined_output
        }
        return result
    
    def _forward_md_only(self):
        """MD-only 模式的前向传播"""
        # 1. 编码MD数据
        outputs_md = self._safe_se_md_call(
            x=self.MDSampler.x,
            pos=self.MDSampler.pos,
            edge_index=(self.MDSampler.edge_index if hasattr(self.MDSampler, "edge_index") else None),
            edge_attr=(self.MDSampler.edge_attr if hasattr(self.MDSampler, "edge_attr") else None),
            batch=(self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None)
        )

        
        # 2. 直接使用 outputs_md 作为输入到 pred_head
        # outputs_md 是字典，包含 'local_se' 和 'global_se'
        local_se = outputs_md.get('local_se')  # [N, H]
        global_se = outputs_md.get('global_se')  # [B, H]
        
        # 使用 global_se 作为 fused representation
        if global_se is not None:
            fused = global_se
        else:
            # 如果没有 global_se，从 local_se 聚合
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None
            if batch_idx is not None and local_se is not None:
                fused = global_mean_pool(local_se, batch_idx)
            elif local_se is not None:
                fused = local_se.mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("SE output contains neither local_se nor global_se")
        
        # 3. 通过 prediction head
        token_direct, global_direct, token_uq, global_uq = self.pred_head(fused.unsqueeze(1))
        
        # 4. 构建输出
        pred_raw = {}
        if global_direct is not None:
            if global_direct.dim() > 1 and global_direct.size(-1) == 1:
                pred_raw["prediction_direct"] = global_direct.view(global_direct.shape[0])
            else:
                pred_raw["prediction_direct"] = global_direct
        if global_uq is not None:
            pred_raw["prediction_uq"] = global_uq
        
        # 5. UQ 后处理
        try:
            uq_out = self.uq(dict(pred_raw)) if pred_raw else {}
        except Exception as e:
            uq_out = dict(pred_raw)
            uq_out["uq_error"] = str(e)
        
        # 6. 合并输出
        combined_output = self.merge_outputs(pred_raw, uq_out)
        
        result = {
            "outputs_md": outputs_md,
            "PreHead": {
                "token_direct": token_direct,
                "global_direct": global_direct,
                "token_uq": token_uq,
                "global_uq": global_uq
            },
            "UQ_out": uq_out,
            "final_output": combined_output
        }
        return result
    
    def _forward_early_fusion(self):
        """Early Fusion 模式的前向传播：简单拼接QM和MD特征"""
        # 1. 编码 QM 数据
        outputs_qm = self.se_qm(
            x=self.QMExtrator.x,
            pos=self.QMExtrator.pos,
            edge_index=(self.QMExtrator.edge_index if hasattr(self.QMExtrator, "edge_index") else None),
            edge_attr=(self.QMExtrator.edge_attr if hasattr(self.QMExtrator, "edge_attr") else None),
            batch=(self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None)
        )
        
        # 2. 编码 MD 数据
        outputs_md = self._safe_se_md_call(
            x=self.MDSampler.x,
            pos=self.MDSampler.pos,
            edge_index=(self.MDSampler.edge_index if hasattr(self.MDSampler, "edge_index") else None),
            edge_attr=(self.MDSampler.edge_attr if hasattr(self.MDSampler, "edge_attr") else None),
            batch=(self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None)
        )
        
        # 3. 获取全局特征
        qm_global = outputs_qm.get('global_se')  # [B, H]
        md_global = outputs_md.get('global_se')  # [B, H]
        
        # 如果没有 global_se，从 local_se 聚合
        if qm_global is None:
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None
            if batch_idx is not None and outputs_qm.get('local_se') is not None:
                qm_global = global_mean_pool(outputs_qm.get('local_se'), batch_idx)
            elif outputs_qm.get('local_se') is not None:
                qm_global = outputs_qm.get('local_se').mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("QM SE output contains neither local_se nor global_se")
        
        if md_global is None:
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None
            if batch_idx is not None and outputs_md.get('local_se') is not None:
                md_global = global_mean_pool(outputs_md.get('local_se'), batch_idx)
            elif outputs_md.get('local_se') is not None:
                md_global = outputs_md.get('local_se').mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("MD SE output contains neither local_se nor global_se")
        
        # 4. Early Fusion: 直接拼接特征（这是与 DUW 的关键区别）
        concatenated = torch.cat([qm_global, md_global], dim=-1)  # [B, 2H]
        
        # 5. 通过 Early Fusion Layer
        fused = self.early_fusion_layer(concatenated)  # [B, H]
        
        # 6. 通过 Prediction Head
        token_direct, global_direct, token_uq, global_uq = self.pred_head(fused.unsqueeze(1))
        
        # 7. 构建输出
        pred_raw = {}
        if global_direct is not None:
            if global_direct.dim() > 1 and global_direct.size(-1) == 1:
                pred_raw["prediction_direct"] = global_direct.view(global_direct.shape[0])
            else:
                pred_raw["prediction_direct"] = global_direct
        if global_uq is not None:
            pred_raw["prediction_uq"] = global_uq
        
        # 8. UQ 后处理
        try:
            uq_out = self.uq(dict(pred_raw)) if pred_raw else {}
        except Exception as e:
            uq_out = dict(pred_raw)
            uq_out["uq_error"] = str(e)
        
        # 9. 合并输出
        combined_output = self.merge_outputs(pred_raw, uq_out)
        
        result = {
            "outputs_qm": outputs_qm,
            "outputs_md": outputs_md,
            "concatenated": concatenated,
            "fused": fused,
            "PreHead": {
                "token_direct": token_direct,
                "global_direct": global_direct,
                "token_uq": token_uq,
                "global_uq": global_uq
            },
            "UQ_out": uq_out,
            "final_output": combined_output
        }
        return result
    
    def _forward_late_fusion(self):
        """Late Fusion 模式的前向传播：分别通过 QM 和 MD 预测，然后平均"""
        # 1. 编码 QM 数据
        outputs_qm = self.se_qm(
            x=self.QMExtrator.x,
            pos=self.QMExtrator.pos,
            edge_index=(self.QMExtrator.edge_index if hasattr(self.QMExtrator, "edge_index") else None),
            edge_attr=(self.QMExtrator.edge_attr if hasattr(self.QMExtrator, "edge_attr") else None),
            batch=(self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None)
        )
        
        # 2. 编码 MD 数据
        outputs_md = self._safe_se_md_call(
            x=self.MDSampler.x,
            pos=self.MDSampler.pos,
            edge_index=(self.MDSampler.edge_index if hasattr(self.MDSampler, "edge_index") else None),
            edge_attr=(self.MDSampler.edge_attr if hasattr(self.MDSampler, "edge_attr") else None),
            batch=(self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None)
        )
        
        # 3. 获取 QM 全局特征
        qm_global = outputs_qm.get('global_se')  # [B, H]
        if qm_global is None:
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None
            if batch_idx is not None and outputs_qm.get('local_se') is not None:
                qm_global = global_mean_pool(outputs_qm.get('local_se'), batch_idx)
            elif outputs_qm.get('local_se') is not None:
                qm_global = outputs_qm.get('local_se').mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("QM SE output contains neither local_se nor global_se")
        
        # 4. 获取 MD 全局特征
        md_global = outputs_md.get('global_se')  # [B, H]
        if md_global is None:
            from torch_geometric.nn import global_mean_pool
            batch_idx = self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None
            if batch_idx is not None and outputs_md.get('local_se') is not None:
                md_global = global_mean_pool(outputs_md.get('local_se'), batch_idx)
            elif outputs_md.get('local_se') is not None:
                md_global = outputs_md.get('local_se').mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("MD SE output contains neither local_se nor global_se")
        
        # 5. QM 通过 pred_head 得到预测
        token_direct_qm, global_direct_qm, token_uq_qm, global_uq_qm = self.pred_head(qm_global.unsqueeze(1))
        
        # 6. MD 通过 pred_head_md 得到预测
        token_direct_md, global_direct_md, token_uq_md, global_uq_md = self.pred_head_md(md_global.unsqueeze(1))
        
        # 7. 构建 QM 输出
        pred_raw_qm = {}
        if global_direct_qm is not None:
            if global_direct_qm.dim() > 1 and global_direct_qm.size(-1) == 1:
                pred_raw_qm["prediction_direct"] = global_direct_qm.view(global_direct_qm.shape[0])
            else:
                pred_raw_qm["prediction_direct"] = global_direct_qm
        if global_uq_qm is not None:
            pred_raw_qm["prediction_uq"] = global_uq_qm
        
        # 8. 构建 MD 输出
        pred_raw_md = {}
        if global_direct_md is not None:
            if global_direct_md.dim() > 1 and global_direct_md.size(-1) == 1:
                pred_raw_md["prediction_direct"] = global_direct_md.view(global_direct_md.shape[0])
            else:
                pred_raw_md["prediction_direct"] = global_direct_md
        if global_uq_md is not None:
            pred_raw_md["prediction_uq"] = global_uq_md
        
        # 9. UQ 后处理
        try:
            uq_out_qm = self.uq(dict(pred_raw_qm)) if pred_raw_qm else {}
        except Exception as e:
            uq_out_qm = dict(pred_raw_qm)
            uq_out_qm["uq_error"] = str(e)
        
        try:
            uq_out_md = self.uq(dict(pred_raw_md)) if pred_raw_md else {}
        except Exception as e:
            uq_out_md = dict(pred_raw_md)
            uq_out_md["uq_error"] = str(e)
        
        # 10. 合并 QM 和 MD 输出
        combined_output_qm = self.merge_outputs(pred_raw_qm, uq_out_qm)
        combined_output_md = self.merge_outputs(pred_raw_md, uq_out_md)
        
        # 11. Late Fusion: 平均两个预测
        pred_qm = combined_output_qm.get('final_prediction')
        pred_md = combined_output_md.get('final_prediction')
        
        if pred_qm is not None and pred_md is not None:
            pred_final = (pred_qm + pred_md) / 2.0
        elif pred_qm is not None:
            pred_final = pred_qm
        elif pred_md is not None:
            pred_final = pred_md
        else:
            pred_final = None
        
        # 12. 平均 logvar（如果存在）
        logvar_qm = combined_output_qm.get('final_logvar')
        logvar_md = combined_output_md.get('final_logvar')
        
        if logvar_qm is not None and logvar_md is not None:
            # 方差的平均：var_avg = (var1 + var2) / 2，所以 logvar_avg = log((exp(logvar1) + exp(logvar2)) / 2)
            var_qm = torch.exp(logvar_qm)
            var_md = torch.exp(logvar_md)
            var_avg = (var_qm + var_md) / 2.0
            logvar_final = torch.log(var_avg + 1e-8)
        elif logvar_qm is not None:
            logvar_final = logvar_qm
        elif logvar_md is not None:
            logvar_final = logvar_md
        else:
            logvar_final = None
        
        # 13. 构建融合后的输出
        combined_output_fused = {
            'final_prediction': pred_final,
            'final_logvar': logvar_final,
            'prediction_qm': pred_qm,
            'prediction_md': pred_md,
            'logvar_qm': logvar_qm,
            'logvar_md': logvar_md,
            'used_uq': combined_output_qm.get('used_uq') or combined_output_md.get('used_uq')
        }
        
        result = {
            "outputs_qm": outputs_qm,
            "outputs_md": outputs_md,
            "qm_global": qm_global,
            "md_global": md_global,
            "PreHead_QM": {
                "token_direct": token_direct_qm,
                "global_direct": global_direct_qm,
                "token_uq": token_uq_qm,
                "global_uq": global_uq_qm
            },
            "PreHead_MD": {
                "token_direct": token_direct_md,
                "global_direct": global_direct_md,
                "token_uq": token_uq_md,
                "global_uq": global_uq_md
            },
            "UQ_out_QM": uq_out_qm,
            "UQ_out_MD": uq_out_md,
            "combined_output_QM": combined_output_qm,
            "combined_output_MD": combined_output_md,
            "final_output": combined_output_fused
        }
        return result
    
    def _forward_dual(self):
        """Dual 模式的前向传播（原有的完整流程）"""
        # placeholders for masks (will be set later using batch ids)
        local_mask = None
        global_mask = None
        # initialize mask placeholders to keep static analyzers happy
        mask_for_attention = None
        local_bool_mask = None
        global_bool_mask = None
        # 1. 编码QM和MD数据
        outputs_qm = self.se_qm(
            x=self.QMExtrator.x,
            pos=self.QMExtrator.pos,
            edge_index=(self.QMExtrator.edge_index if hasattr(self.QMExtrator, "edge_index") else None),
            edge_attr=(self.QMExtrator.edge_attr if hasattr(self.QMExtrator, "edge_attr") else None),
            batch=(self.QMExtrator.batch if hasattr(self.QMExtrator, "batch") else None)
        )
        outputs_md = self._safe_se_md_call(
            x=self.MDSampler.x,
            pos=self.MDSampler.pos,
            edge_index=(self.MDSampler.edge_index if hasattr(self.MDSampler, "edge_index") else None),
            edge_attr=(self.MDSampler.edge_attr if hasattr(self.MDSampler, "edge_attr") else None),
            batch=(self.MDSampler.batch if hasattr(self.MDSampler, "batch") else None)
        )

        # 2. 融合QM和MD特征
        # 不要在这里强制将 Fusion 置为 eval()，这会破坏 model.train()/model.eval() 的一致性
        # 确保 outputs_qm/outputs_md 包含 batch 信息，以便 Fusion 能做按图池化
        try:
            if isinstance(outputs_qm, dict) and 'batch' not in outputs_qm and hasattr(self.QMExtrator, 'batch'):
                outputs_qm['batch'] = self.QMExtrator.batch
        except Exception:
            pass
        try:
            if isinstance(outputs_md, dict) and 'batch' not in outputs_md and hasattr(self.MDSampler, 'batch'):
                outputs_md['batch'] = self.MDSampler.batch
        except Exception:
            pass

        fusion_out = self.fusion(outputs_qm, outputs_md)
        # collect fusion stats for debug
        fusion_stats = fusion_out.get('fusion_stats', {}) if isinstance(fusion_out, dict) else {}

        # Cast fusion outputs to float32 (avoid fp16 under/overflow in downstream UE/attention)
        try:
            if isinstance(fusion_out, dict):
                lf = fusion_out.get('local_fusion', None)
                if isinstance(lf, torch.Tensor) and lf.dtype in (torch.float16, torch.bfloat16):
                    fusion_out['local_fusion'] = lf.float()
                gf = fusion_out.get('global_fusion', None)
                if isinstance(gf, torch.Tensor) and gf.dtype in (torch.float16, torch.bfloat16):
                    fusion_out['global_fusion'] = gf.float()
        except Exception:
            pass

        # 3. 准备 batch id 列表，用于按 ID/索引对齐预计算的 masks
        current_batch_ids = self._get_current_batch_ids()

        # 4. 计算不确定性（兼容新版UE键名）
        qm_UE = self.qm_UE()
        md_UE = self.md_UE()
        # 兼容字段：
        #  - local matrix (canonical): "local_uncertainty_matrix"（兼容旧名："uncertainty_matrix"）；
        #    若存在 raw 版本（local_uncertainty_matrix_raw），优先用于 Attention/权重
        #  - global vec   (canonical): "global_logvar_vector"（兼容旧名："global_uncertainty_vector"）
        def _first_present(d: dict, keys):
            for k in keys:
                v = d.get(k, None)
                if v is not None:
                    return v
            return None

        # Prefer raw matrices when available to drive attention with unnormalized signals
        # Support new UE key names: prefer raw, then norm, then legacy keys
        self.qm_mat = _first_present(qm_UE, ["local_uncertainty_matrix_raw", "local_uncertainty_matrix_norm", "local_uncertainty_matrix", "uncertainty_matrix"])  # [B, L]
        self.qm_global = _first_present(qm_UE, ["global_logvar_vector", "global_uncertainty_vector"])  # [B, 1]
        self.md_mat = _first_present(md_UE, ["local_uncertainty_matrix_raw", "local_uncertainty_matrix_norm", "local_uncertainty_matrix", "uncertainty_matrix"])  # [B, L]
        self.md_global = _first_present(md_UE, ["global_logvar_vector", "global_uncertainty_vector"])  # [B, 1]

        # 运行时健壮性检查（便于定位 H5/UE 输出异常）
        if self.qm_mat is None:
            print("[DUW::Module] Warning: QM UE 缺少 'local_uncertainty_matrix'（或兼容名）")
        if self.md_mat is None:
            print("[DUW::Module] Warning: MD UE 缺少 'local_uncertainty_matrix'（或兼容名）")
        if self.qm_global is None:
            print("[DUW::Module] Warning: QM UE 缺少 'global_logvar_vector'（或兼容名）")
        if self.md_global is None:
            print("[DUW::Module] Warning: MD UE 缺少 'global_logvar_vector'（或兼容名）")

        # Ensure keys lists exist for alignment (added by UE modules)
        qm_keys = qm_UE.get("keys", None)
        md_keys = md_UE.get("keys", None)

        # If both matrices exist, align them by common keys to avoid batch-size mismatch
        if self.qm_mat is not None and self.md_mat is not None and qm_keys is not None and md_keys is not None:
            # find intersection and indices
            qm_key_to_idx = {k: i for i, k in enumerate(qm_keys)}
            md_key_to_idx = {k: i for i, k in enumerate(md_keys)}
            common_keys = [k for k in qm_keys if k in md_key_to_idx]

            if len(common_keys) == 0:
                raise RuntimeError("No common keys between QM and MD uncertainty estimators; cannot align batches.")

            qm_indices = torch.tensor([qm_key_to_idx[k] for k in common_keys], dtype=torch.long, device=self.device)
            md_indices = torch.tensor([md_key_to_idx[k] for k in common_keys], dtype=torch.long, device=self.device)

            # Guard against out-of-range indices before CUDA index_select (avoid device-side asserts)
            try:
                n_qm = int(self.qm_mat.size(0)) if isinstance(self.qm_mat, torch.Tensor) else 0
                n_md = int(self.md_mat.size(0)) if isinstance(self.md_mat, torch.Tensor) else 0
            except Exception:
                n_qm = n_md = 0

            if n_qm > 0 and n_md > 0 and qm_indices.numel() == md_indices.numel():
                keep_pos = []
                q_list = qm_indices.detach().cpu().tolist()
                m_list = md_indices.detach().cpu().tolist()
                for i, (qi, mi) in enumerate(zip(q_list, m_list)):
                    if 0 <= int(qi) < n_qm and 0 <= int(mi) < n_md:
                        keep_pos.append(i)
                if len(keep_pos) < len(common_keys):
                    import logging
                    logging.warning(
                        "[DUW::_forward_dual] Filtering out %d/%d out-of-range UE indices (qm<%d, md<%d).",
                        len(common_keys) - len(keep_pos), len(common_keys), n_qm, n_md,
                    )
                    if keep_pos:
                        keep_tensor = torch.tensor(keep_pos, dtype=torch.long, device=self.device)
                        qm_indices = torch.index_select(qm_indices, 0, keep_tensor)
                        md_indices = torch.index_select(md_indices, 0, keep_tensor)
                        common_keys = [common_keys[i] for i in keep_pos]
                    else:
                        raise RuntimeError("All UE indices are out-of-range after bounds check; cannot align UE matrices.")

            # select aligned rows (after bounds-checked indices)
            qm_sel = torch.index_select(self.qm_mat.to(self.device), 0, qm_indices)
            md_sel = torch.index_select(self.md_mat.to(self.device), 0, md_indices)

            # 对齐当前批次的分子 ID，只保留与当前 batch 匹配的 key（避免输出批次被放大）
            # Use precomputed current_batch_ids when available; do not overwrite it here.
            batch_ids_for_alignment = current_batch_ids or self._get_current_batch_ids()

            if batch_ids_for_alignment:
                # 保留 batch id 原始顺序，建立从 UE 子集行 -> batch 行号映射
                batch_id_list = list(batch_ids_for_alignment)
                batch_id_set = set(batch_id_list)
                keep_positions = []      # positions in common_keys
                keep_batch_pos = []      # corresponding positions in current batch
                for i, key in enumerate(common_keys):
                    sk = str(key)
                    if sk in batch_id_set:
                        keep_positions.append(i)
                        try:
                            pos_b = batch_id_list.index(sk)
                            keep_batch_pos.append(pos_b)
                        except ValueError:
                            pass
                if keep_positions:
                    keep_idx_tensor = torch.tensor(keep_positions, dtype=torch.long, device=self.device)
                    qm_indices = torch.index_select(qm_indices, 0, keep_idx_tensor)
                    md_indices = torch.index_select(md_indices, 0, keep_idx_tensor)
                    qm_sel = torch.index_select(qm_sel, 0, keep_idx_tensor)
                    md_sel = torch.index_select(md_sel, 0, keep_idx_tensor)
                    common_keys = [common_keys[i] for i in keep_positions]
                    # 记录与 batch 行号的对应关系（用于后续 mask 重对齐）
                    try:
                        self.batch_keep_positions = torch.tensor(keep_batch_pos, dtype=torch.long)
                    except Exception:
                        self.batch_keep_positions = None
                else:
                    # 无交集：保留全部 common_keys（不设置 batch_keep_positions）
                    self.batch_keep_positions = None

            # 5. 计算不确定性权重矩阵 on aligned subset
            def _san_vec(v: torch.Tensor, ceiling: float = 1e6, floor: float = -1e6):
                if v is None or not isinstance(v, torch.Tensor):
                    return v, 0
                nonfinite = (~torch.isfinite(v)).sum().item()
                v = torch.nan_to_num(v, nan=0.0, posinf=ceiling, neginf=floor)
                try:
                    v = v.clamp(min=floor, max=ceiling)
                except Exception:
                    v = torch.clamp(v, min=floor, max=ceiling)
                return v, int(nonfinite)

            qm_sel, qm_nonfinite = _san_vec(qm_sel)
            md_sel, md_nonfinite = _san_vec(md_sel)
            # track nonfinite counts for debug
            self.qm_nonfinite = int(qm_nonfinite)
            self.md_nonfinite = int(md_nonfinite)
            self.weight_mat = torch.einsum("bi,bj->bij", qm_sel, md_sel)

            # global vectors aligned
            if self.qm_global is not None and self.md_global is not None:
                qm_glob_sel = torch.index_select(self.qm_global.to(self.device), 0, qm_indices)
                md_glob_sel = torch.index_select(self.md_global.to(self.device), 0, md_indices)
                self.global_weight_mat = qm_glob_sel * md_glob_sel
            else:
                self.global_weight_mat = None

            # store selected versions for downstream use (masking/CoAttention)
            self.qm_mat_aligned = qm_sel
            self.md_mat_aligned = md_sel
            self.common_keys = common_keys
            # keep indices for aligning masks later
            self.qm_indices = qm_indices
            self.md_indices = md_indices
        else:
            # fallback: try elementwise einsum and let PyTorch raise if shapes mismatch
            try:
                def _san_vec_v(v: torch.Tensor, ceiling: float = 1e6, floor: float = -1e6):
                    if v is None or not isinstance(v, torch.Tensor):
                        return v, 0
                    nonfinite = (~torch.isfinite(v)).sum().item()
                    v = torch.nan_to_num(v, nan=0.0, posinf=ceiling, neginf=floor)
                    try:
                        v = v.clamp(min=floor, max=ceiling)
                    except Exception:
                        v = torch.clamp(v, min=floor, max=ceiling)
                    return v, int(nonfinite)

                qm_vec, qnf = _san_vec_v(self.qm_mat)
                md_vec, mnf = _san_vec_v(self.md_mat)
                self.qm_nonfinite = int(qnf)
                self.md_nonfinite = int(mnf)
                self.weight_mat = torch.einsum("bi,bj->bij", qm_vec, md_vec)
            except Exception as e:
                raise
            if self.qm_global is not None and self.md_global is not None:
                self.global_weight_mat = self.qm_global * self.md_global
            else:
                self.global_weight_mat = None

        # 6. 【新增】将mask应用到不确定性权重上
        # 获取bool mask用于过滤不确定性权重（按当前 batch 的 ids 对齐）
        local_bool_mask = None
        global_bool_mask = None
        try:
            if current_batch_ids:
                local_bool_mask = self.Localmask.get_bool_mask(batch_idx=current_batch_ids)
                global_bool_mask = self.Globalmask.get_bool_mask(batch_idx=current_batch_ids)
            else:
                local_bool_mask = self.Localmask.get_bool_mask()
                global_bool_mask = self.Globalmask.get_bool_mask()
        except Exception:
            # on any failure, fall back to safe full-mask retrieval
            try:
                local_bool_mask = self.Localmask.get_bool_mask()
            except Exception:
                local_bool_mask = None
            try:
                global_bool_mask = self.Globalmask.get_bool_mask()
            except Exception:
                global_bool_mask = None

        # 使用 batch_keep_positions 对掩码进行子集裁剪（与 UE 对齐后的 batch 顺序一致）
        if isinstance(local_bool_mask, torch.Tensor) and hasattr(self, 'batch_keep_positions') and isinstance(self.batch_keep_positions, torch.Tensor):
            idx_b = self.batch_keep_positions.to(local_bool_mask.device)
            if idx_b.numel() > 0 and idx_b.max().item() < local_bool_mask.size(0):
                try:
                    local_bool_mask = local_bool_mask.index_select(0, idx_b)
                except Exception:
                    pass
        if isinstance(global_bool_mask, torch.Tensor) and hasattr(self, 'batch_keep_positions') and isinstance(self.batch_keep_positions, torch.Tensor):
            idx_b = self.batch_keep_positions.to(global_bool_mask.device)
            if idx_b.numel() > 0 and idx_b.max().item() < global_bool_mask.size(0):
                try:
                    global_bool_mask = global_bool_mask.index_select(0, idx_b)
                except Exception:
                    pass

        # 方法1: 重新梳理本地掩码到权重矩阵（行/列掩码），并加入“全零行”保护
        # 思路：从 Localmask 的 token 级 bool 掩码（[B_full, L_local]）出发，
        # 分别 resize 到 Lq、Lk 得到行/列掩码，然后施加到 weight_mat。
        weight_mat_masked = self.weight_mat.clone()
        try:
            B_w, Lq, Lk = int(weight_mat_masked.size(0)), int(weight_mat_masked.size(1)), int(weight_mat_masked.size(2))
        except Exception:
            B_w = Lq = Lk = 0

        if isinstance(local_bool_mask, torch.Tensor) and local_bool_mask.dim() >= 2 and B_w > 0:
            # token 级掩码 [B_full, L_local]
            token_mask = local_bool_mask.any(dim=-1).float()

            # 已通过 batch_keep_positions 对齐，不再使用 qm_indices

            # resize 工具：优先用最近邻插值到目标长度；失败则用截断/补 1 回退
            def _resize_mask(mask_bl: torch.Tensor, target_len: int) -> torch.Tensor:
                if mask_bl is None:
                    return torch.ones((B_w, target_len), device=weight_mat_masked.device, dtype=torch.float32)
                try:
                    inp = mask_bl.to(device=weight_mat_masked.device, dtype=torch.float32).unsqueeze(1)  # [B,1,L]
                    out = torch.nn.functional.interpolate(inp, size=int(target_len), mode='nearest').squeeze(1)
                    return (out > 0.5).float()
                except Exception:
                    mm = mask_bl.to(device=weight_mat_masked.device, dtype=torch.float32)
                    if mm.size(1) == target_len:
                        return mm
                    if mm.size(1) > target_len:
                        return mm[:, :target_len]
                    pad_cols = target_len - mm.size(1)
                    pad = torch.ones((mm.size(0), pad_cols), device=mm.device, dtype=mm.dtype)
                    return torch.cat([mm, pad], dim=1)

            row_mask = _resize_mask(token_mask, Lq)  # [B_w, Lq]
            col_mask = _resize_mask(token_mask, Lk)  # [B_w, Lk]

            # 应用行/列掩码（乘法），并在后续加“全零行”保护
            weight_mat_masked = weight_mat_masked * row_mask.unsqueeze(2) * col_mask.unsqueeze(1)

            # 保护：若某行（某个 query 的所有 key）被掩到 0，则为该行提供安全的 fallback
            eps = 1e-8
            row_sum = weight_mat_masked.abs().sum(dim=-1, keepdim=True)  # [B_w, Lq, 1]
            zero_row_mask = (row_sum <= eps)  # [B_w, Lq, 1]
            if zero_row_mask.any():
                # 计算每行的安全 fallback（使用原始 weight_mat 的行均值，若仍为0则使用 uncert_floor）
                try:
                    orig_row_mean = self.weight_mat.to(weight_mat_masked.device).mean(dim=-1, keepdim=True)  # [B_w, Lq, 1]
                except Exception:
                    orig_row_mean = torch.zeros_like(row_sum, device=weight_mat_masked.device)
                # 若 orig_row_mean 的绝对值太小，则使用 uncert_floor 或更小的 epsilon 作为 fallback
                fallback_val = max(float(getattr(self, 'uncert_floor', 1e-6)), 1e-6)
                # 若 orig_row_mean 太大（有溢出风险或受 inf 替换影响），使用 fallback 替代
                max_allowed = float(getattr(self, 'uncert_ceiling', 1e3))
                large_mask = (orig_row_mean.abs() > max_allowed)
                safe_row = torch.where(orig_row_mean.abs() <= eps, torch.ones_like(orig_row_mean) * fallback_val, orig_row_mean)
                safe_row = torch.where(large_mask, torch.ones_like(safe_row) * fallback_val, safe_row)
                safe_row = safe_row.expand(-1, -1, Lk)
                zero_row = zero_row_mask.expand(-1, -1, Lk)
                weight_mat_masked = torch.where(zero_row, safe_row, weight_mat_masked)
                # 记录恢复比例，供 debug 输出
                restored_frac = float(zero_row_mask.float().mean().cpu().item())
                setattr(self, 'weight_mask_zero_row_restored_frac', restored_frac)

        # 方法2: 也可以mask全局权重
        global_weight_mat_masked = None
        if self.global_weight_mat is not None:
            try:
                global_weight_mat_masked = self.global_weight_mat.clone()
            except Exception:
                global_weight_mat_masked = self.global_weight_mat

        if global_weight_mat_masked is not None and isinstance(global_bool_mask, torch.Tensor) and global_bool_mask.dim() >= 1:
            global_mask_scalar = global_bool_mask.any(dim=-1).float()  # 已对齐子集后的 [B_w]

            try:
                global_mask_scalar = global_mask_scalar.to(global_weight_mat_masked.device)
            except Exception:
                pass
            # 保持形状一致：[B,1] * [B,1]
            if global_weight_mat_masked.dim() == 2 and global_weight_mat_masked.size(-1) == 1:
                global_weight_mat_masked = global_weight_mat_masked * global_mask_scalar.unsqueeze(-1)
            else:
                global_weight_mat_masked = global_weight_mat_masked * global_mask_scalar

        # 为 CoAttention 准备要传入的 mask 与 uncert_weight
        # Always derive a token-level boolean mask [B, L] for attention to avoid
        # ambiguity when LocalMask.get_mask returns a [B, L, H] additive mask.
        if current_batch_ids:
            mb = self.Localmask.get_bool_mask(batch_idx=current_batch_ids)
        else:
            mb = self.Localmask.get_bool_mask()

        # mb may be [B, L, H] or [B, L]; convert to token-level bool [B, L]
        if isinstance(mb, torch.Tensor):
            if mb.dim() == 3:
                mask_for_attention = mb.any(dim=-1)
            elif mb.dim() == 2:
                mask_for_attention = mb
            else:
                mask_for_attention = None
        else:
            mask_for_attention = None

        # If we aligned QM/MD UE to a subset of keys, align the attention mask to the same subset
        if hasattr(self, 'batch_keep_positions') and isinstance(self.batch_keep_positions, torch.Tensor) and mask_for_attention is not None:
            try:
                idx = self.batch_keep_positions.to(mask_for_attention.device)
                if idx.numel() > 0 and idx.max().item() < mask_for_attention.size(0):
                    mask_for_attention = mask_for_attention.index_select(0, idx)
            except Exception:
                pass

        # 注意：不要进一步对齐 global_fusion，因为 qm_mat_aligned 和 md_mat_aligned 
        # 已经在前面通过 index_select 对齐过了。
        # 如果再次对齐 global_fusion，会导致维度不匹配（global_fusion 被缩小，但 qm_mat_aligned 保持原样）
        # 应该保持 global_fusion 与 qm_mat_aligned/md_mat_aligned 的 batch 维度一致
        
        # 对已 mask 的权重做最终清洗：替换非有限、裁剪到安全范围，记录非有限计数
        try:
            if isinstance(weight_mat_masked, torch.Tensor):
                nonfinite_masked = (~torch.isfinite(weight_mat_masked)).sum().item()
                weight_mat_masked = torch.nan_to_num(weight_mat_masked, nan=0.0, posinf=1e6, neginf=-1e6)
                try:
                    weight_mat_masked = weight_mat_masked.clamp(min=-1e6, max=1e6)
                except Exception:
                    weight_mat_masked = torch.clamp(weight_mat_masked, min=-1e6, max=1e6)
                self.weight_mat_masked_nonfinite = int(nonfinite_masked)
        except Exception:
            self.weight_mat_masked_nonfinite = 0

        # 将已 mask 的不确定性权重与 mask 传入 attention，交由 attention 内部做对齐
        CoAttention = self.UncertaintyCoAttention(
            local_fusion=fusion_out["local_fusion"],
            global_fusion=fusion_out["global_fusion"],  # 使用原始的 global_fusion，不再次对齐
            qm_uncert=(getattr(self, 'qm_mat_aligned', self.qm_mat)),
            md_uncert=(getattr(self, 'md_mat_aligned', self.md_mat)),
            uncert_weight=weight_mat_masked,
            mask=mask_for_attention,
        )

        # Run gated aggregator on CoAttention outputs (use local attns for pooling)
        local_qm = CoAttention["local"]["qm"]
        local_md = CoAttention["local"]["md"]
        attn_qm2md = CoAttention["local"].get("attn_qm2md")
        attn_md2qm = CoAttention["local"].get("attn_md2qm")
        global_qm = CoAttention["global"].get("qm")
        global_md = CoAttention["global"].get("md")

        # 统一所有输入的 batch 维度，避免 cat 时出现 B 不一致；仅做 padding/扩展，不截断
        def _pad_batch_dim(tensor, target_B):
            if tensor is None:
                return None
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.size(0) == target_B:
                return tensor
            if tensor.size(0) < target_B:
                pad_shape = (target_B - tensor.size(0),) + tuple(tensor.shape[1:])
                pad_tensor = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
                return torch.cat([tensor, pad_tensor], dim=0)
            # 若 tensor B 更大，则直接返回（后续 target_B 会基于所有张量的 max 计算）
            return tensor

        batch_candidates = []
        for t in [local_qm, local_md]:
            if isinstance(t, torch.Tensor):
                batch_candidates.append(t.size(0))
        for t in [global_qm, global_md, attn_qm2md, attn_md2qm]:
            if isinstance(t, torch.Tensor):
                if t.dim() >= 1:
                    batch_candidates.append(t.size(0))

        target_batch = max(batch_candidates) if batch_candidates else 1

        local_qm = _pad_batch_dim(local_qm, target_batch)
        local_md = _pad_batch_dim(local_md, target_batch)
        if isinstance(attn_qm2md, torch.Tensor):
            attn_qm2md = _pad_batch_dim(attn_qm2md, target_batch)
        if isinstance(attn_md2qm, torch.Tensor):
            attn_md2qm = _pad_batch_dim(attn_md2qm, target_batch)
        if isinstance(global_qm, torch.Tensor):
            global_qm = _pad_batch_dim(global_qm, target_batch)
        if isinstance(global_md, torch.Tensor):
            global_md = _pad_batch_dim(global_md, target_batch)

        ga_out = self.gated_aggregator(
            local_qm=local_qm,
            local_md=local_md,
            global_qm=global_qm,
            global_md=global_md,
            attn_qm2md=attn_qm2md,
            attn_md2qm=attn_md2qm,
        )
        ga_fused_stats = ga_out.get('fused_stats', {}) if isinstance(ga_out, dict) else {}

        # Run PredictionHead but keep its outputs separate from gate_out
        fused = ga_out.get("fused_representation")
        # PredictionHead expects [B, N, H]; treat fused as single-token local_feats
        token_direct, global_direct, token_uq, global_uq = self.pred_head(fused.unsqueeze(1))

        # Collect raw prediction outputs into a separate dict (do NOT write into ga_out)
        pred_raw = {}
        if global_direct is not None:
            # ensure scalar shape [B] if global_direct_dim==1
            if global_direct.dim() > 1 and global_direct.size(-1) == 1:
                pred_raw["prediction_direct"] = global_direct.view(global_direct.shape[0])
            else:
                pred_raw["prediction_direct"] = global_direct
        if global_uq is not None:
            pred_raw["prediction_uq"] = global_uq

        # Post-process prediction outputs with UncertaintyQuantification into a separate dict
        try:
            uq_out = self.uq(dict(pred_raw)) if pred_raw else {}
        except Exception as e:
            uq_out = dict(pred_raw)
            uq_out["uq_error"] = str(e)

        # Merge outputs into a single structured combined_output
        combined_output = self.merge_outputs(pred_raw, uq_out)

        # ================= 调试信息（可选）=================
        debug_metrics = {}
        if self.debug:
            try:
                # 1) fused 表示层统计
                if isinstance(fused, torch.Tensor):
                    debug_metrics['fused_mean'] = float(fused.mean().detach().cpu().item())
                    debug_metrics['fused_std_mean_dim'] = float(fused.std(dim=0).mean().detach().cpu().item())
                # 2) 注意力熵（局部双向）
                def _entropy(t: torch.Tensor) -> torch.Tensor:
                    p = t.clamp(min=1e-12, max=1.0)
                    return -(p * p.log()).sum(dim=-1)  # [B, Nq]
                if isinstance(attn_qm2md, torch.Tensor):
                    ent_qm2md = _entropy(attn_qm2md.detach())
                    debug_metrics['attn_qm2md_entropy_mean'] = float(ent_qm2md.mean().cpu().item())
                if isinstance(attn_md2qm, torch.Tensor):
                    ent_md2qm = _entropy(attn_md2qm.detach())
                    debug_metrics['attn_md2qm_entropy_mean'] = float(ent_md2qm.mean().cpu().item())
                # 3) 权重矩阵统计（原始与掩码后）
                if isinstance(self.weight_mat, torch.Tensor):
                    wm = self.weight_mat.detach()
                    debug_metrics['weight_mat_mean'] = float(wm.mean().cpu().item())
                    debug_metrics['weight_mat_std'] = float(wm.std().cpu().item())
                if isinstance(weight_mat_masked, torch.Tensor):
                    wmm = weight_mat_masked.detach()
                    debug_metrics['weight_mat_masked_mean'] = float(wmm.mean().cpu().item())
                    debug_metrics['weight_mat_masked_std'] = float(wmm.std().cpu().item())
                # qm/md nonfinite counts
                try:
                    debug_metrics['qm_nonfinite'] = int(getattr(self, 'qm_nonfinite', 0))
                    debug_metrics['md_nonfinite'] = int(getattr(self, 'md_nonfinite', 0))
                    debug_metrics['weight_mat_clamped'] = bool(getattr(self, 'weight_mat_clamped', False))
                except Exception:
                    pass
                # masked weight sanitization counts
                try:
                    debug_metrics['weight_mat_masked_nonfinite'] = int(getattr(self, 'weight_mat_masked_nonfinite', 0))
                except Exception:
                    pass
                # restored zero-row fraction due to mask
                try:
                    debug_metrics['weight_mask_zero_row_restored_frac'] = float(getattr(self, 'weight_mask_zero_row_restored_frac', 0.0))
                except Exception:
                    pass
                # Fusion input/output std diagnostics
                try:
                    debug_metrics['fusion_in_std_qm'] = float(fusion_stats.get('in_std_qm', 0.0))
                    debug_metrics['fusion_in_std_md'] = float(fusion_stats.get('in_std_md', 0.0))
                    debug_metrics['fusion_out_std'] = float(fusion_stats.get('out_std', 0.0))
                    debug_metrics['fusion_scale_used'] = float(fusion_stats.get('scale_used', 1.0))
                except Exception:
                    pass
                try:
                    debug_metrics['ga_fused_in_std'] = float(ga_fused_stats.get('preserve_scale_in_std', 0.0))
                    debug_metrics['ga_fused_out_std'] = float(ga_fused_stats.get('preserve_scale_out_std', 0.0))
                    debug_metrics['ga_fused_scale_used'] = float(ga_fused_stats.get('preserve_scale_scale_used', 1.0))
                except Exception:
                    pass
                # attention uncert handling config
                try:
                    debug_metrics['uncert_attn_mode'] = str(getattr(self, 'uncert_attn_mode', 'mult'))
                    debug_metrics['uncert_attn_scale'] = float(getattr(self, 'uncert_attn_scale', 1.0))
                    debug_metrics['uncert_attn_eps'] = float(getattr(self, 'uncert_attn_eps', 1e-6))
                except Exception:
                    pass
                # nonfinite counts on masked weights and clamped flag
                try:
                    debug_metrics['weight_mat_masked_nonfinite'] = int(getattr(self, 'weight_mat_masked_nonfinite', 0))
                except Exception:
                    pass
                # 4) 预测分支差异
                if 'prediction_direct' in combined_output and 'final_prediction' in combined_output:
                    pd = combined_output['prediction_direct']
                    fp = combined_output['final_prediction']
                    if isinstance(pd, torch.Tensor) and isinstance(fp, torch.Tensor):
                        diff = (pd.view(-1) - fp.view(-1)).abs().mean().detach().cpu().item()
                        debug_metrics['direct_vs_final_abs_mean_diff'] = float(diff)
                        # also print per-branch variance
                        try:
                            debug_metrics['prediction_direct_std'] = float(pd.view(-1).std().detach().cpu().item())
                        except Exception:
                            pass
                        try:
                            debug_metrics['final_prediction_std'] = float(fp.view(-1).std().detach().cpu().item())
                        except Exception:
                            pass
                # 5) 全零行比例（mask 后权重）
                if isinstance(weight_mat_masked, torch.Tensor):
                    zero_rows = (weight_mat_masked.abs().sum(dim=-1) <= 1e-8).float().mean().cpu().item()
                    debug_metrics['masked_zero_row_fraction'] = float(zero_rows)
            except Exception as e:
                debug_metrics['debug_error'] = str(e)
            # 打印（简洁）
            try:
                print('[DUW::debug]', {k: round(v, 6) if isinstance(v, float) else v for k, v in debug_metrics.items()})
            except Exception:
                pass

        result = {
            "outputs_qm": outputs_qm,
            "outputs_md": outputs_md,
            "fusion_out": fusion_out,
            "uncertainty": {
                "qm": {
                    "qm_mat": self.qm_mat,
                    "qm_global": self.qm_global,
                },
                "md": {
                    "md_mat": self.md_mat,
                    "md_global": self.md_global,
                },
                "weight_mat": self.weight_mat,  # 原始权重
                "weight_mat_masked": weight_mat_masked,  # 应用mask后的权重
                "global_weight_mat": self.global_weight_mat,  # 原始全局权重
                "global_weight_mat_masked": global_weight_mat_masked  # 应用mask后的全局权重
            },
            "masks": {
                "local_mask_for_attn": (mask_for_attention if isinstance(mask_for_attention, torch.Tensor) else None),
                "local_bool": local_bool_mask,
                "global_bool": global_bool_mask,
                "global_mask_scalar": (global_bool_mask.any(dim=-1).float() if isinstance(global_bool_mask, torch.Tensor) and getattr(global_bool_mask, 'dim', lambda: 0)() >= 1 else None)
            },
            "CoAttention": CoAttention,
            "gate_out": ga_out,
            "PreHead": {
                "token_direct": token_direct,
                "global_direct": global_direct,
                "token_uq": token_uq,
                "global_uq": global_uq
            },
            "UQ_out": uq_out,
            "final_output": combined_output,  # 改为 final_output 以保持一致性
            "debug_metrics": debug_metrics if self.debug else {}
        }
        return result



