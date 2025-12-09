import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    多任务鲁棒物理约束损失：

    目标与场景
    - QM: 图级 Binding_Affinity (高置信度)
    - MD: 节点/图级 feature_atoms_adaptability (潜在噪声，需自适应弱化)

    组成
    1) 亲和力主损失 (QM MSE)
     2) 力分布对齐 (MD 层次):
         - 局部(节点级)：|F| 与 adaptability 的标准化分布对齐（按图 z-score）
         - 全局(图级)：Mean(|F|) 与 Mean(adaptability) 的标准化值对齐（按 batch z-score）
       两者采用置信度加权 + 同方差不确定性(learnable log_var)自动下调不可靠帧
    3) 物理约束：零均值力、力平滑(沿边)

    说明
    - 若无真实力/能量标签，仅从能量预测对位置求导得到预测力
    - 若 energy_pred 与 pos 无梯度路径，力相关正则会退化为 fallback（零/谐振），相当于自动不启用
    - 兼容仅传入 batch_md 的旧用法：QM 目标优先取 batch_qm.y；若缺失则回退到 batch_md.y
    """

    def __init__(self,
                 w_energy: float = 1.0,
                 # 层次对齐权重
                 w_md_local: float = 0.1,
                 w_md_global: float = 0.1,
                 # 物理约束
                 w_force_zero: float = 0.0,
                 w_force_smooth: float = 0.0,
                 # 兼容旧接口: w_forces (映射到 w_md_local)
                 w_forces: float | None = None,
                 # 其他
                 force_fallback: str = 'harmonic',
                 compute_forces_from_energy: bool = True,
                 force_reduction: str = 'sum',
                 energy_agg: str = 'sum',
                 # 置信度与稳定性
                 conf_temperature: float = 1.0,
                 eps: float = 1e-8):
        """
        Args:
            w_energy: QM 亲和力主损失权重
            w_md_local: MD 节点级力分布对齐权重
            w_md_global: MD 图级力分布对齐权重
            w_force_zero: 零均值力约束权重
            w_force_smooth: 力平滑(沿边差分)权重
            force_fallback: 力计算失败时的回退策略('zero'|'harmonic')
            compute_forces_from_energy: 是否从能量对 pos 求导得到力
            force_reduction: 保留参数以兼容原实现
            energy_agg: 当 energy_target 为节点级时聚合到图级的方式('sum'|'mean')
            conf_temperature: 置信度温度，>1 更平滑，<1 更尖锐
            eps: 数值稳定性
        """
        super().__init__()
        self.w_energy = w_energy
        self.w_md_local = w_md_local
        self.w_md_global = w_md_global
        self.w_force_zero = w_force_zero
        self.w_force_smooth = w_force_smooth
        # 旧接口兼容逻辑：若传入 w_forces 且局部/全局权重都为0，则将其作为局部对齐权重
        if w_forces is not None:
            if w_md_local == 0.0 and w_md_global == 0.0:
                self.w_md_local = w_forces
            # 存储原始 w_forces 方便旧测试访问
            self.w_forces = w_forces
        else:
            # 若未提供旧参数，定义 w_forces 供旧测试读取 (使用局部权重代表)
            self.w_forces = self.w_md_local
        self.force_fallback = force_fallback
        self.compute_forces_from_energy = compute_forces_from_energy
        if energy_agg not in ('sum', 'mean'):
            raise ValueError("energy_agg must be 'sum' or 'mean'")
        self.energy_agg = energy_agg
        self.force_reduction = force_reduction
        self.conf_temperature = conf_temperature
        self.eps = eps

        # 同方差不确定性：自动调整 MD 两级损失的权重，弱化不可靠帧
        self.md_log_var_local = nn.Parameter(torch.tensor(0.0))
        self.md_log_var_global = nn.Parameter(torch.tensor(0.0))

    def forward(self, ga_out: dict, batch_md, batch_qm=None, energy_pred: torch.Tensor = None, energy_logvar: torch.Tensor = None, device: torch.device = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            ga_out: 模型输出字典，需包含 'prediction' 或 'fused_representation'
            batch_md: MD 图数据(含 pos, batch, edge_index, y=feature_atoms_adaptability)
            batch_qm: QM 图数据(含 y=Binding_Affinity)，可选
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        if device is None:
            device = self._get_device(ga_out)
        total_loss = torch.tensor(0.0, device=device)

        # 1) 亲和力主损失
        # Prefer energy_pred explicitly passed by the training loop (final_output branch).
        if energy_pred is None:
            energy_pred = self.extract_energy_prediction(ga_out)
        # ensure tensor on correct device
        if energy_pred is not None:
            energy_pred = energy_pred.to(device)

        # 优先从 QM 取 Binding_Affinity；无则兼容旧逻辑从 MD 取
        energy_target = None
        if batch_qm is not None:
            energy_target = self._extract_target(batch_qm, ['y', 'energy', 'targets'])
        if energy_target is None:
            energy_target = self._extract_target(batch_md, ['y', 'energy', 'targets'])
        if energy_target is not None:
            energy_target = energy_target.to(device)
            # 若为节点级，将其聚合到图级
            energy_target = self._process_energy_target(energy_target, batch_qm if batch_qm is not None else batch_md)


        if self.w_energy > 0:
            # If a predictive log-variance is provided use NLL; otherwise use MSE
            if energy_logvar is not None:
                # move logvar to device and compute gaussian NLL
                try:
                    energy_logvar = energy_logvar.to(device)
                    energy_loss = self.compute_energy_nll(energy_pred, energy_logvar, energy_target)
                except Exception:
                    energy_loss = self.compute_energy_loss(energy_pred, energy_target)
            else:
                energy_loss = self.compute_energy_loss(energy_pred, energy_target)
            # 兼容旧训练循环使用 'energy' 键统计
            losses['energy'] = energy_loss
            losses['qm_energy'] = energy_loss  # 提供更语义化名称，同时保留旧键
            total_loss += self.w_energy * energy_loss

        # 2) 预测力(由能量对 MD 位置求导)
        predicted_forces = None
        md_pos = None
        md_batch_vec = None
        if self.compute_forces_from_energy:
            try:
                md_pos = self._extract_positions(batch_md).to(device)
                md_batch_vec = self._extract_batch_vector(batch_md).to(device)
                predicted_forces = self.compute_predicted_forces(energy_pred, md_pos, md_batch_vec)
            except Exception:
                predicted_forces = None

        # 3) MD 层次对齐（需要 predicted_forces 和 MD 的 y=feature_atoms_adaptability）
        md_y = self._extract_target(batch_md, ['y'])
        if md_y is not None:
            md_y = md_y.to(device)

        # 3.1 节点级(local)
        if self.w_md_local > 0 and predicted_forces is not None and md_y is not None:
            try:
                local_loss, local_conf = self._local_alignment_loss(predicted_forces, md_y, md_batch_vec)
                # 不确定性加权: 0.5 * exp(-s) * L + 0.5 * s
                local_weight = torch.exp(-self.md_log_var_local)
                local_loss_u = 0.5 * local_weight * local_loss + 0.5 * self.md_log_var_local
                losses['md_local_align'] = local_loss
                losses['md_local_align_u'] = local_loss_u
                losses['md_local_conf_mean'] = local_conf.mean()
                total_loss += self.w_md_local * local_loss_u
            except Exception:
                pass

        # 3.2 图级(global)
        if self.w_md_global > 0 and predicted_forces is not None and md_y is not None:
            try:
                global_loss, global_conf = self._global_alignment_loss(predicted_forces, md_y, md_batch_vec)
                global_weight = torch.exp(-self.md_log_var_global)
                global_loss_u = 0.5 * global_weight * global_loss + 0.5 * self.md_log_var_global
                losses['md_global_align'] = global_loss
                losses['md_global_align_u'] = global_loss_u
                losses['md_global_conf_mean'] = global_conf.mean()
                total_loss += self.w_md_global * global_loss_u
            except Exception:
                pass

        # 4) 物理约束
        if self.w_force_zero > 0 and predicted_forces is not None and md_batch_vec is not None:
            zero_loss = self._zero_mean_force_loss(predicted_forces, md_batch_vec)
            losses['force_zero'] = zero_loss
            total_loss += self.w_force_zero * zero_loss

        if self.w_force_smooth > 0 and predicted_forces is not None:
            try:
                edge_index = self._extract_edge_index(batch_md)
                smooth_loss = self._force_smoothness_loss(predicted_forces, edge_index)
                losses['force_smooth'] = smooth_loss
                total_loss += self.w_force_smooth * smooth_loss
            except Exception:
                pass

        return total_loss, losses

    def extract_energy_prediction(self, ga_out: dict) -> torch.Tensor:
        """从GatedAggregator输出提取能量预测"""
        if 'prediction' in ga_out:
            return ga_out['prediction'].view(-1)
        elif 'fused_representation' in ga_out:
            return ga_out['fused_representation'].mean(dim=-1)
        else:
            raise ValueError('ga_out must contain "prediction" or "fused_representation"')

    def compute_energy_loss(self, energy_pred: torch.Tensor, energy_target: torch.Tensor) -> torch.Tensor:
        """计算能量损失"""
        # 确保形状一致
        if energy_pred.shape != energy_target.shape:
            # 允许 energy_target 为 [B,1] 的情况
            if energy_target.dim() == 2 and energy_target.shape[1] == 1 and energy_target.shape[0] == energy_pred.shape[0]:
                energy_target = energy_target.squeeze(-1)
            # 如果目标是单个标量（例如 [1] 或标量tensor），则广播到预测形状
            elif energy_target.numel() == 1:
                try:
                    energy_target = energy_target.view(-1).expand_as(energy_pred)
                except Exception:
                    energy_target = energy_pred.new_full(energy_pred.shape, float(energy_target.item()))
            else:
                raise ValueError(f"Energy prediction shape {energy_pred.shape} does not match target shape {energy_target.shape}")
        return F.mse_loss(energy_pred, energy_target, reduction='mean')

    def compute_energy_nll(self, energy_pred: torch.Tensor, energy_logvar: torch.Tensor, energy_target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian NLL using predicted log-variance per-sample.

        NLL = 0.5 * (logvar + (target - pred)^2 / exp(logvar)) + 0.5*log(2*pi)
        We return mean NLL across samples.
        """
        # align shapes
        if energy_logvar.shape != energy_pred.shape:
            try:
                energy_logvar = energy_logvar.view_as(energy_pred)
            except Exception:
                # fallback to broadcasting
                energy_logvar = energy_logvar.expand_as(energy_pred)
        var = torch.exp(energy_logvar)
        sq = (energy_target - energy_pred).pow(2)
        nll = 0.5 * (energy_logvar + sq / (var + self.eps))
        # add constant term 0.5*log(2*pi)
        nll = nll + 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793, device=nll.device))
        return nll.mean()

    def compute_predicted_forces(self, energy_pred: torch.Tensor, positions: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """从能量预测计算力: F = -∂E/∂x"""
        if not self.compute_forces_from_energy:
            raise RuntimeError("Force computation is disabled. Set compute_forces_from_energy=True")
        # If energy_pred is not connected to the positions via autograd graph,
        # we cannot compute dE/dx. In that case, fall back to the configured strategy.
        try:
            if not getattr(energy_pred, 'requires_grad', False):
                return self._fallback_forces(energy_pred, positions)

            # 确保 positions 启用梯度跟踪
            if not positions.requires_grad:
                positions = positions.clone().detach().requires_grad_(True)
            
            total_energy = energy_pred.sum()
            try:
                grads = torch.autograd.grad(
                    outputs=total_energy,
                    inputs=positions,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                return self._fallback_forces(energy_pred, positions)

            if grads is None:
                return self._fallback_forces(energy_pred, positions)
            return -grads
        except Exception:
            return self._fallback_forces(energy_pred, positions)

    def compute_force_loss(self, predicted_forces: torch.Tensor, target_forces: torch.Tensor) -> torch.Tensor:
        """计算力损失"""
        return F.mse_loss(predicted_forces, target_forces, reduction=self.force_reduction)

    # ========= 层次对齐与物理约束 ========= #
    def _local_alignment_loss(self, predicted_forces: torch.Tensor, md_y: torch.Tensor, batch_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
    节点级：对齐 |F| 与 adaptability 的标准化分布（按图 z-score）。
        - 若 md_y 为 [N] 或 [N,1]，视为节点级；否则尝试广播。
        返回: (loss, node_confidence)
        """
        N = predicted_forces.size(0)
        fmag = predicted_forces.norm(dim=1)  # [N]
        if md_y.dim() == 2 and md_y.size(1) == 1:
            md_y = md_y.view(-1)
        if md_y.numel() != N:
            # 尝试广播为节点级
            if md_y.numel() == 1:
                md_y = md_y.view(1).expand(N)
            else:
                # 形状不匹配则退出
                raise ValueError("MD y shape not compatible for local alignment")

        # 按图标准化（z-score），减少绝对量纲影响
        fmag_norm = self._normalize_per_graph(fmag, batch_vec)
        adapt_norm = self._normalize_per_graph(md_y, batch_vec)

        # 置信度: 基于标准化信号，温度缩放的 sigmoid 到 [0,1]
        node_conf = self._confidence_from_signal(adapt_norm)
        # 节点加权 MSE
        diff = (fmag_norm - adapt_norm).pow(2)
        loss = (node_conf * diff).sum() / (node_conf.sum() + self.eps)
        return loss, node_conf.detach()

    def _global_alignment_loss(self, predicted_forces: torch.Tensor, md_y: torch.Tensor, batch_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        图级：对齐 Mean(|F|) 与 Mean(adaptability)
        返回: (loss, graph_confidence)
        """
        fmag = predicted_forces.norm(dim=1)  # [N]
        bmax = int(batch_vec.max().item()) + 1
        # 聚合为图级均值
        fmean = self._scatter_mean(fmag, batch_vec, bmax)  # [B]

        # 将 md_y 也变为图级均值（若其为节点级）或校验图级
        if md_y.dim() == 2 and md_y.size(1) == 1:
            md_y = md_y.view(-1)
        if md_y.numel() == fmag.numel():
            ymean = self._scatter_mean(md_y, batch_vec, bmax)
        elif md_y.numel() == bmax:
            ymean = md_y
        elif md_y.numel() == 1:
            ymean = md_y.view(1).expand(bmax)
        else:
            raise ValueError("MD y shape not compatible for global alignment")

        # 图级标准化（batch 维度 z-score）
        fmean_mu = fmean.mean()
        fmean_std = fmean.std()
        ymean_mu = ymean.mean()
        ymean_std = ymean.std()
        # 若 std≈0，退化为零中心（避免除零）
        if torch.isfinite(fmean_std) and fmean_std > self.eps:
            fmean_norm = (fmean - fmean_mu) / (fmean_std + self.eps)
        else:
            fmean_norm = fmean - fmean_mu
        if torch.isfinite(ymean_std) and ymean_std > self.eps:
            ymean_norm = (ymean - ymean_mu) / (ymean_std + self.eps)
        else:
            ymean_norm = ymean - ymean_mu

        # 图级置信度（来自标准化的 ymean_norm），温度缩放
        gconf = self._confidence_from_signal(ymean_norm)
        diff = (fmean_norm - ymean_norm).pow(2)
        loss = (gconf * diff).sum() / (gconf.sum() + self.eps)
        return loss, gconf.detach()

    def _zero_mean_force_loss(self, predicted_forces: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """零均值力约束: 每个图的平均力向量接近 0"""
        bmax = int(batch_vec.max().item()) + 1
        idx = batch_vec.view(-1).to(device=predicted_forces.device, dtype=torch.long)
        ones = torch.ones_like(idx, dtype=predicted_forces.dtype)
        pred_sum = torch.zeros(bmax, 3, device=predicted_forces.device).index_add(0, idx, predicted_forces)
        counts = torch.zeros(bmax, device=predicted_forces.device).index_add(0, idx, ones).clamp(min=1.0).view(-1,1)
        pred_mean = pred_sum / counts
        return (pred_mean.pow(2).sum(dim=1).mean())

    def _force_smoothness_loss(self, predicted_forces: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """沿边的力差分平滑: mean(||F_i - F_j||^2)"""
        if edge_index is None:
            return predicted_forces.new_tensor(0.0)
        i, j = edge_index[0], edge_index[1]
        diff = predicted_forces[i] - predicted_forces[j]
        return (diff.pow(2).sum(dim=1).mean())

    # ========= 实用函数 ========= #
    def _fallback_forces(self, energy_pred: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """处理梯度为None的情况"""
        if self.force_fallback == 'zero':
            return torch.zeros_like(positions)
        elif self.force_fallback == 'harmonic':
            k = energy_pred.abs().mean() * 1e-6
            return -k * positions
        else:
            return torch.zeros_like(positions)

    def _extract_target(self, batch_md, keys: list) -> torch.Tensor:
        """从batch_md中提取目标字段"""
        for key in keys:
            if hasattr(batch_md, key):
                return getattr(batch_md, key)
            if isinstance(batch_md, dict) and key in batch_md:
                return batch_md[key]
        return None

    def _extract_positions(self, batch_md) -> torch.Tensor:
        """提取位置信息"""
        if hasattr(batch_md, 'pos'):
            return batch_md.pos
        elif isinstance(batch_md, dict) and 'pos' in batch_md:
            return batch_md['pos']
        else:
            raise ValueError("Cannot find position data")

    def _extract_batch_vector(self, batch_md) -> torch.Tensor:
        """提取批次向量"""
        if hasattr(batch_md, 'batch'):
            return batch_md.batch
        elif isinstance(batch_md, dict) and 'batch' in batch_md:
            return batch_md['batch']
        else:
            raise ValueError("Cannot find batch vector")

    def _extract_edge_index(self, batch_md) -> torch.Tensor:
        """提取边索引 edge_index (2, E)"""
        if hasattr(batch_md, 'edge_index'):
            return batch_md.edge_index
        elif isinstance(batch_md, dict) and 'edge_index' in batch_md:
            return batch_md['edge_index']
        else:
            return None

    def _get_device(self, ga_out: dict) -> torch.device:
        """获取计算设备"""
        for key in ['prediction', 'fused_representation']:
            if key in ga_out:
                return ga_out[key].device
        return torch.device('cpu')

    def _process_energy_target(self, energy_target: torch.Tensor, batch_md) -> torch.Tensor:
        """处理能量目标的维度"""
        batch_vec = self._extract_batch_vector(batch_md)

        # normalize shapes: allow energy_target to be [N,1]
        if energy_target is None:
            return None
        if energy_target.dim() == 2 and energy_target.shape[1] == 1:
            energy_target = energy_target.squeeze(-1)

        # expected graph count
        if batch_vec is None:
            raise ValueError("batch vector is missing from batch_md")
        bmax = int(batch_vec.max().item()) + 1

        # If already graph-level, return
        if energy_target.shape[0] == bmax:
            return energy_target

        # If atomic-level (one value per atom/node), aggregate to graph-level
        if energy_target.numel() == batch_vec.numel():
            # ensure index is long and on same device as energy_target
            idx = batch_vec.to(device=energy_target.device, dtype=torch.long)
            sums = torch.zeros(bmax, device=energy_target.device).scatter_add_(0, idx, energy_target.view(-1))
            if self.energy_agg == 'sum':
                return sums
            else:
                # mean: need counts per graph
                ones = torch.ones((energy_target.numel(),), device=energy_target.device)
                counts = torch.zeros(bmax, device=energy_target.device).scatter_add_(0, idx, ones)
                counts = counts.clamp(min=1.0)
                return sums / counts

        # Last resort: shapes incompatible — raise informative error
        raise ValueError(
            f"Cannot align energy_target shape {tuple(energy_target.shape)} with batch (num_graphs={bmax}, batch_vec_len={batch_vec.numel()})"
        )

    def _scatter_mean(self, values: torch.Tensor, batch_vec: torch.Tensor, bmax: int) -> torch.Tensor:
        idx = batch_vec.to(device=values.device, dtype=torch.long)
        sums = torch.zeros(bmax, device=values.device).index_add(0, idx, values.view(-1))
        ones = torch.ones_like(values.view(-1))
        counts = torch.zeros(bmax, device=values.device).index_add(0, idx, ones)
        counts = counts.clamp(min=1.0)
        return sums / counts

    def _normalize_per_graph(self, values: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """按图做标准化（z-score）：(x - mean_g(x)) / (std_g(x) + eps)"""
        bmax = int(batch_vec.max().item()) + 1
        idx = batch_vec.to(device=values.device, dtype=torch.long)
        means = self._scatter_mean(values, batch_vec, bmax)  # [B]
        # 计算每图方差
        # E[x^2] - (E[x])^2
        vals_sq = values.view(-1) * values.view(-1)
        sum_sq = torch.zeros(bmax, device=values.device).index_add(0, idx, vals_sq)
        ones = torch.ones_like(values.view(-1))
        counts = torch.zeros(bmax, device=values.device).index_add(0, idx, ones).clamp(min=1.0)
        ex2 = sum_sq / counts
        var = (ex2 - means.pow(2)).clamp(min=0.0)
        std = torch.sqrt(var + self.eps)
        means_per_node = means[idx]
        std_per_node = std[idx]
        # 若某图 std≈0，仅做去均值
        z = (values - means_per_node)
        z = torch.where(std_per_node > self.eps, z / (std_per_node + self.eps), z)
        return z

    def _confidence_from_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        基于标准化信号的置信度映射：先做 z-score（按 batch），再用温度缩放的 sigmoid 压到 (0,1)。
        conf = sigmoid( z / T )，T=conf_temperature。
        """
        s = signal
        mu = s.mean()
        std = s.std()
        if torch.isfinite(std) and std > self.eps:
            z = (s - mu) / (std + self.eps)
        else:
            z = s - mu
        T = self.conf_temperature if self.conf_temperature is not None and self.conf_temperature > 0 else 1.0
        conf = torch.sigmoid(z / T)
        return conf
