import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(x):
    """统一转换成 float32 1D tensor，不做标准化"""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return torch.tensor([0.0], dtype=torch.float32)
    if x.ndim == 0:
        x = np.array([float(x)], dtype=np.float32)
    return torch.tensor(x, dtype=torch.float32)


class QMUncertaintyEstimator(nn.Module):
    """
    【改进版 QM UE】
    ——输出两个关键量：
        1) local_uncertainty_norm  : [B, L]（z-score，局部token级别）
        2) molecule_total_logvar   : [B, 1]（全局不确定性 log-variance）

    特点：
    - 局部不确定性基于原子属性方差（或波动）
    - 标准化处理保证尺度稳定，不会因数值过大/过小导致 Gate 不收敛
    - 输出 logvar 用于 Gate logits 调节，非常稳定
    """

    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path

        # 用于把标量不确定性投射到 Gate 的 hidden_dim 维
        self.proj_dim = 32
        self.logvar_proj = nn.Linear(1, self.proj_dim)
        self.local_norm = nn.LayerNorm(1)

    def forward(self):
        qm_results = {}
        local_list = []         # 标准化后的局部 UE
        local_raw_list = []     # 原始局部 UE（不标准化）
        global_list = []
        _keys_order = []

        with h5py.File(self.h5_path, "r") as f:
            for mol_id in f.keys():
                mol_group = f[mol_id]

                # ----------- (1) 提取原子属性作为局部不确定性 -----------
                atom_group = mol_group["atom_properties"]
                if "atom_properties_values" in atom_group:
                    atom_values = atom_group["atom_properties_values"][()].flatten()
                else:
                    atom_values = np.array([0.0], dtype=np.float32)

                atom_unc = to_tensor(atom_values)  # shape [L]

                #**局部标准化（数值稳定，不让 UE 失控）**
                mu = atom_unc.mean()
                std = atom_unc.std()
                if torch.isfinite(std) and std > 1e-6:
                    local_norm = (atom_unc - mu) / (std + 1e-6)
                else:
                    local_norm = atom_unc.new_zeros(atom_unc.shape)

                # ----------- (2) 图级不确定性（对 atom_unc 求方差） -----------
                # raw variance
                mol_var = atom_unc.var().unsqueeze(0)  # [1]

                # 转为 logvar，clamp 避免过大/过小
                mol_logvar = torch.log(mol_var + 1e-6)
                mol_logvar = mol_logvar.clamp(-5.0, 5.0)   # 强稳定性

                # ----------- 记录 -----------
                qm_results[mol_id] = {
                    "local_uncertainty": atom_unc,           # 原始局部 UE
                    "local_uncertainty_norm": local_norm,    # 标准化局部 UE (给 attention)
                    "molecule_total_logvar": mol_logvar      # 全局 logvar (给 Gate)
                }

                local_list.append(local_norm)
                local_raw_list.append(atom_unc)
                global_list.append(mol_logvar)
                _keys_order.append(mol_id)

        # ----------- 拼 batch 返回 -----------

        # 局部 pad 到统一长度 —— 明确区分标准化与原始两个矩阵，避免覆盖
        if local_list:
            max_len = max([x.size(0) for x in local_list])
            padded_locals_norm = [F.pad(x, (0, max_len - x.size(0))) for x in local_list]
            qm_results["local_uncertainty_matrix_norm"] = torch.stack(padded_locals_norm)   # [B, L]（标准化）

        # 原始（未标准化）矩阵，供需要 raw 的注意力/权重使用
        if local_raw_list:
            max_len_raw = max([x.size(0) for x in local_raw_list])
            padded_locals_raw = [F.pad(x, (0, max_len_raw - x.size(0))) for x in local_raw_list]
            qm_results["local_uncertainty_matrix_raw"] = torch.stack(padded_locals_raw)  # [B, L]

        if len(_keys_order) > 0:
            qm_results["keys"] = list(_keys_order)

        if global_list:
            qm_results["global_logvar_vector"] = torch.stack(global_list)  # [B,1]

        return qm_results
