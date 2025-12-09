import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(x):
    """统一转换成 float32 的 1D tensor，不做标准化"""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return torch.tensor([0.0], dtype=torch.float32)
    if x.ndim == 0:
        x = np.array([float(x)], dtype=np.float32)
    return torch.tensor(x, dtype=torch.float32)


class MDUncertaintyEstimator(nn.Module):
    """
    【改进版 MD UE】
    ——输出（forward，给主网络用）：
        * local_uncertainty_norm   : [L_md]   标准化（z-score）MD 序列不确定性（给 Attention Bias）
        * molecule_total_logvar    : [1]      全局 log-variance（给 Gate 约束）
    """

    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path

    def _extract_series(self, md_group, key):
        """安全加载 HDF5 序列（可能不存在）"""
        if key in md_group:
            return to_tensor(md_group[key][()])
        return torch.tensor([0.0], dtype=torch.float32)

    def _normalize(self, x):
        """稳定标准化（z-score）：(x-mean)/std；若 std≈0，则返回零向量。"""
        mu = x.mean()
        std = x.std()
        if torch.isfinite(std) and std > 1e-6:
            return (x - mu) / (std + 1e-6)
        # 退化情形：无波动，返回全零避免数值爆炸
        return x.new_zeros(x.shape)

    # ---------- 主网络仍然调用这个，保持不变 ----------
    def forward(self):
        md_results = {}
        local_list = []
        global_list = []
        _keys_order = []

        with h5py.File(self.h5_path, "r") as f:
            for mol_id in f.keys():
                md_group = f[mol_id]

                # (1) 收集四种 MD 序列
                series_list = []
                for key in [
                    "frames_interaction_energy",
                    "frames_rmsd_ligand",
                    "frames_distance",
                    "frames_bSASA"
                ]:
                    seq = self._extract_series(md_group, key)
                    series_list.append(seq)

                # 拼为一个长序列 [L_md]
                local_raw = torch.cat(series_list)

                # (2) 局部不确定性（标准化）
                local_norm = self._normalize(local_raw)

                # (3) 全局不确定性（logvar，非常稳定，给主网络用）
                raw_var = local_raw.var().unsqueeze(0)  # [1]
                logvar = torch.log(raw_var + 1e-6).clamp(-5.0, 5.0)

                md_results[mol_id] = {
                    "local_uncertainty": local_raw,          # 原始序列 UE
                    "local_uncertainty_norm": local_norm,    # 标准化 UE（用于 attention）
                    "molecule_total_logvar": logvar          # 全局 log-variance（给 gate）
                }

                local_list.append(local_raw)
                global_list.append(logvar)
                _keys_order.append(mol_id)

        # (4) 统一 padding 成 batch
        if local_list:
            max_len = max(x.size(0) for x in local_list)
            padded_locals = [F.pad(x, (0, max_len - x.size(0))) for x in local_list]
            md_results["local_uncertainty_matrix"] = torch.stack(padded_locals)
            md_results["keys"] = list(_keys_order)

        if global_list:
            md_results["global_logvar_vector"] = torch.stack(global_list)

        return md_results

    # ---------- 新接口：专门给可视化用（不 clamp） ----------
    def export_for_plot(
        self,
        max_mols: int = 200,
        downsample_len: int = 20,
        shuffle: bool = True,
        seed: int = 42,
        use_norm: bool = True,
    ):
        """
        导出适合画 MD 不确定性那几张图的数据（不会在主网络里自动调用）

        返回:
            {
              "mol_ids":         [B]
              "global_var_raw":  [B]  每个分子的原始方差（不截断）
              "global_std_raw":  [B]  sqrt(var_raw)
              "local_tokens":    [N]  拼接后的 token 序列（可画右上角直方图）
              "traj_matrix":     [B,L] 下采样后的轨迹（画折线 + 统计图）
            }
        """
        rng = np.random.RandomState(seed)
        mol_ids = []
        global_var_raw = []
        global_std_raw = []
        local_tokens_all = []
        traj_rows = []

        with h5py.File(self.h5_path, "r") as f:
            all_ids = list(f.keys())
            if shuffle:
                rng.shuffle(all_ids)

            for mol_id in all_ids[:max_mols]:
                md_group = f[mol_id]

                # 与 forward 同样的四条序列
                series_list = []
                for key in [
                    "frames_interaction_energy",
                    "frames_rmsd_ligand",
                    "frames_distance",
                    "frames_bSASA"
                ]:
                    seq = self._extract_series(md_group, key)
                    series_list.append(seq)

                local_raw = torch.cat(series_list)           # [L]
                local_norm = self._normalize(local_raw)

                # 这里 **不做 clamp**，只是普通方差
                var_raw = float(local_raw.var(unbiased=False).item())
                std_raw = float(np.sqrt(var_raw + 1e-6))

                seq = local_norm if use_norm else local_raw
                seq_np = seq.detach().cpu().numpy()

                mol_ids.append(str(mol_id))
                global_var_raw.append(var_raw)
                global_std_raw.append(std_raw)
                local_tokens_all.append(seq_np)

                # 下采样轨迹
                L = seq_np.shape[0]
                if L <= downsample_len:
                    traj = seq_np
                else:
                    idx = np.linspace(0, L - 1, downsample_len).astype(int)
                    traj = seq_np[idx]
                traj_rows.append(traj)

        if not mol_ids:
            raise ValueError("export_for_plot: HDF5 中没有可用的 MD 轨迹。")

        # 对齐轨迹长度
        max_L = max(len(r) for r in traj_rows)
        final_L = min(max_L, downsample_len)
        padded_rows = []
        for r in traj_rows:
            if len(r) >= final_L:
                padded_rows.append(r[:final_L])
            else:
                pad = np.full(final_L - len(r), np.nan, dtype=np.float32)
                padded_rows.append(np.concatenate([r, pad], axis=0))

        traj_matrix = np.stack(padded_rows, axis=0)

        return {
            "mol_ids": mol_ids,
            "global_var_raw": np.asarray(global_var_raw, dtype=np.float64),
            "global_std_raw": np.asarray(global_std_raw, dtype=np.float64),
            "local_tokens": np.concatenate(local_tokens_all, axis=0).astype(np.float64),
            "traj_matrix": traj_matrix.astype(np.float64),
        }

    def from_dict(self, md_dict):
        """
        根据单个分子的 MD 序列字典，计算局部/全局不确定性。
        兼容 compare_qm_md_uncertainty.py 的用法，返回字段包括：
            - local_uncertainty          : 原始拼接序列 [L]
            - local_uncertainty_norm     : z-score 标准化后的序列 [L]
            - molecule_total_logvar      : 标量 log-variance [1]
            - molecule_total_uncertainty : 标量 sigma = sqrt(exp(logvar)) [1]
        md_dict 预期包含的 key：
            "frames_interaction_energy", "frames_rmsd_ligand",
            "frames_distance", "frames_bSASA"
        """
        series_list = []

        for key in [
            "frames_interaction_energy",
            "frames_rmsd_ligand",
            "frames_distance",
            "frames_bSASA",
        ]:
            if key in md_dict:
                # 从 numpy / list 转为 1D float32 tensor
                vals = md_dict[key]
                seq = torch.as_tensor(vals, dtype=torch.float32).flatten()
            else:
                # 缺这个通道时，用 0 占位，避免 cat 报错
                seq = torch.zeros(1, dtype=torch.float32)
            series_list.append(seq)

        # 拼成一个长序列 [L]
        local_raw = torch.cat(series_list, dim=0)

        # 标准化（用你类里的 _normalize）
        local_norm = self._normalize(local_raw)

        # 1) 原始方差（无偏差=False，和 export_for_plot 一致）
        var_raw = local_raw.var(unbiased=False)

        # 2) 全局不确定性：直接用 std = sqrt(var)
        sigma = torch.sqrt(var_raw + 1e-6)

        # 3) logvar 只做记录，不再 clamp（可选）
        logvar = torch.log(var_raw + 1e-6)

        return {
            "local_uncertainty": local_raw,
            "local_uncertainty_norm": local_norm,
            "molecule_total_logvar": logvar,
            "molecule_total_uncertainty": sigma,
        }
