import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import logging


# from Fusion import *


# ---------------- Energy ----------------
def Energy(qmh5_file, local_target_len=128, hidden_dim=128, device: torch.device = None):
    qm_energy_local_list, qm_energy_global_list = [], []
    all_mol_values, all_atom_values = [], []

    with h5py.File(qmh5_file, "r") as f:
        for mol_id in f.keys():
            mol_group = f[mol_id]
            mol_props = mol_group["mol_properties"]
            atom_vals = mol_group["atom_properties"]["atom_properties_values"][()].flatten()
            all_mol_values.append([mol_props[key][()] for key in mol_props.keys()])
            all_atom_values.append(atom_vals)

    # create tensors on specified device when possible
    if device is None:
        mol_vals_tensor = torch.tensor(all_mol_values, dtype=torch.float32)
    else:
        mol_vals_tensor = torch.tensor(all_mol_values, dtype=torch.float32, device=device)

    mol_mu, mol_sigma = mol_vals_tensor.mean(0), mol_vals_tensor.std(0) + 1e-8

    max_len = max(len(a) for a in all_atom_values)
    if device is None:
        atom_padded = torch.zeros((len(all_atom_values), max_len), dtype=torch.float32)
    else:
        atom_padded = torch.zeros((len(all_atom_values), max_len), dtype=torch.float32, device=device)

    for i, a in enumerate(all_atom_values):
        if device is None:
            at = torch.tensor(a, dtype=torch.float32)
            atom_padded[i, :len(a)] = at
        else:
            at = torch.tensor(a, dtype=torch.float32, device=device)
            atom_padded[i, :len(a)] = at

    atom_mu, atom_sigma = atom_padded.mean(0), atom_padded.std(0) + 1e-8

    with h5py.File(qmh5_file, "r") as f:
        for mol_id in f.keys():
            mol_group = f[mol_id]
            mol_props = mol_group["mol_properties"]
            atom_vals = mol_group["atom_properties"]["atom_properties_values"][()].flatten()

            if device is None:
                mol_vals = torch.tensor([mol_props[key][()] for key in mol_props.keys()], dtype=torch.float32)
            else:
                mol_vals = torch.tensor([mol_props[key][()] for key in mol_props.keys()], dtype=torch.float32,
                                        device=device)

            global_energy = ((mol_vals - mol_mu) / mol_sigma).mean().repeat(hidden_dim)
            qm_energy_global_list.append(global_energy)

            if device is None:
                atom_vals_t = torch.tensor(atom_vals, dtype=torch.float32)
            else:
                atom_vals_t = torch.tensor(atom_vals, dtype=torch.float32, device=device)
            atom_std = (atom_vals_t - atom_mu[:len(atom_vals)]) / atom_sigma[:len(atom_vals)]
            if len(atom_std) < local_target_len:
                atom_std = F.pad(atom_std, (0, local_target_len - len(atom_std)))
            else:
                atom_std = F.adaptive_avg_pool1d(atom_std.unsqueeze(0), local_target_len).squeeze(0)
            qm_energy_local_list.append(atom_std.unsqueeze(-1).repeat(1, hidden_dim))

    return torch.stack(qm_energy_local_list), torch.stack(qm_energy_global_list)


# ---------------- Mask Utilities ----------------
def bool_to_additive_mask(bool_mask, mask_value=-1e9):
    """
    将 bool mask 转换为 additive mask（用于logits/attention scores）

    Args:
        bool_mask: [B, L] or [B, L, H] bool tensor, True表示保留，False表示mask掉
        mask_value: float, mask掉的位置填充的值（默认-1e9，经过softmax后接近0）

    Returns:
        additive_mask: 同shape的float tensor，保留位置为0，mask位置为mask_value
    """
    # True (1.0) -> 0.0 (保留)
    # False (0.0) -> mask_value (屏蔽)
    return (1.0 - bool_mask.float()) * mask_value


def bool_to_multiplicative_mask(bool_mask):
    """
    将 bool mask 转换为 multiplicative mask（用于特征/权重）

    Args:
        bool_mask: bool tensor, True表示保留，False表示mask掉

    Returns:
        multiplicative_mask: float tensor，保留位置为1.0，mask位置为0.0
    """
    return bool_mask.float()


# ---------------- Local Mask ----------------
class LocalMask(nn.Module):
    def __init__(self, qmh5_file, hidden_dim, local_target_len, energy_threshold=1.0,
                 mask_type='multiplicative', mask_value=-1e9, device: torch.device = None):
        """
        局部掩码模块

        Args:
            qmh5_file: QM数据的h5文件路径
            hidden_dim: 隐藏层维度
            local_target_len: 局部特征长度
            energy_threshold: 能量阈值（<=threshold的保留，>threshold的mask掉）
            mask_type: 'multiplicative' 或 'additive'
                - 'multiplicative': 直接乘在特征上 (feature * mask)
                - 'additive': 加到logits上 (logits + mask)，用于attention机制
            mask_value: additive mask使用的值（默认-1e9）
        """
        super().__init__()
        self.energy_threshold = energy_threshold
        self.hidden_dim = hidden_dim
        self.local_target_len = local_target_len
        self.qmh5_file = qmh5_file
        self.mask_type = mask_type
        self.mask_value = mask_value

        # precompute energy masks (on provided device when possible)
        self.energy_local, _ = Energy(self.qmh5_file, local_target_len=self.local_target_len,
                                      hidden_dim=self.hidden_dim, device=device)
        # Also store ordered mol ids from the HDF5 so callers can index masks by ID
        try:
            with h5py.File(self.qmh5_file, 'r') as fh:
                self.mol_ids = [str(k) for k in fh.keys()]
        except Exception:
            logging.exception("Failed to read mol ids from %s", str(self.qmh5_file))
            # fallback: create numeric row indices
            self.mol_ids = [str(i) for i in range(self.energy_local.size(0))] if isinstance(self.energy_local, torch.Tensor) else []
        # build id -> row index mapping
        try:
            self.id_to_idx = {mid: i for i, mid in enumerate(self.mol_ids)}
        except Exception:
            self.id_to_idx = {}

    def get_bool_mask(self, batch_idx=None):
        """返回bool类型的mask，True表示保留，False表示屏蔽

        Args:
            batch_idx: None or LongTensor/list/ndarray of indices into the precomputed
                energy tensor. If None, returns the full dataset mask (shape [N_mol, L, H]).
                If provided, returns per-batch mask (shape [B, L, H]).
        """
        if batch_idx is None:
            return self.energy_local <= self.energy_threshold

        # Safe indexing: build per-request mask rows while validating indices on CPU
        try:
            N = self.energy_local.shape[0]
            default_row = self.energy_local[0]

            masks = []

            # Case: list/tuple of mol-id strings -> map via id_to_idx
            if isinstance(batch_idx, (list, tuple)) and len(batch_idx) > 0 and isinstance(batch_idx[0], str):
                idx_list = [self.id_to_idx.get(str(x), None) for x in batch_idx]
                bad = []
                for i, maybe in enumerate(idx_list):
                    if maybe is None or not (0 <= int(maybe) < N):
                        bad.append((i, batch_idx[i], maybe))
                        masks.append(default_row)
                    else:
                        masks.append(self.energy_local[int(maybe)])
                if bad:
                    logging.warning("LocalMask: some mol ids not found or out-of-range: %s; using default row for those.", bad)

            else:
                # Try to interpret batch_idx as integer indices (tensor/list/ndarray/scalar)
                try:
                    # convert to CPU python list for validation
                    idx_list = torch.as_tensor(batch_idx, dtype=torch.long).cpu().tolist()
                except Exception:
                    # fallback: single scalar
                    try:
                        single = int(batch_idx)
                        idx_list = [single]
                    except Exception:
                        logging.exception("LocalMask: unable to coerce batch_idx=%s to indices; returning full mask.", str(batch_idx))
                        return self.energy_local <= self.energy_threshold

                # idx_list may be an int (if scalar) or list
                if isinstance(idx_list, int):
                    idx_list = [idx_list]

                bad = []
                for i, v in enumerate(idx_list):
                    try:
                        vi = int(v)
                    except Exception:
                        bad.append((i, v))
                        masks.append(default_row)
                        continue
                    if not (0 <= vi < N):
                        bad.append((i, vi))
                        masks.append(default_row)
                    else:
                        masks.append(self.energy_local[vi])

                if bad:
                    logging.warning("LocalMask: some provided indices are out-of-range or invalid: %s; using default row for those.", bad)

            # Stack masks into a tensor on the same device as energy_local
            if len(masks) == 0:
                # nothing valid -> fallback to full dataset mask
                logging.warning("LocalMask: no valid indices found in batch_idx=%s; returning full mask.", str(batch_idx))
                return self.energy_local <= self.energy_threshold

            mask_tensor = torch.stack(masks, dim=0).to(self.energy_local.device)
            return (mask_tensor <= self.energy_threshold)

        except Exception:
            logging.exception("Failed to index energy_local with batch_idx=%s; returning full mask as fallback.", str(batch_idx))
            return self.energy_local <= self.energy_threshold

    def get_mask(self, batch_idx=None):
        """根据mask_type返回相应格式的mask。支持按 batch_idx 索引已预计算的 mask。"""
        bool_mask = self.get_bool_mask(batch_idx=batch_idx)

        if self.mask_type == 'additive':
            return bool_to_additive_mask(bool_mask, self.mask_value)
        elif self.mask_type == 'multiplicative':
            return bool_to_multiplicative_mask(bool_mask)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def forward(self, local_fusion, batch_idx=None, return_mask=False):
        """
        Args:
            local_fusion: [L, H] 或 [B, L, H] 局部融合特征
            batch_idx: optional indices (list/tensor) selecting which rows from the
                precomputed energy mask to use for this batch. If None and local_fusion
                is a single-sample [L,H], the first entry of the precomputed mask will be used.
            return_mask: 是否返回mask本身

        Returns:
            如果mask_type='multiplicative': 返回masked特征 (local_fusion * mask)
            如果mask_type='additive': 返回mask本身，需要在外部加到logits上
        """
        mask = self.get_mask(batch_idx=batch_idx)

        # Align mask shape to local_fusion
        # local_fusion could be [L,H] or [B,L,H]
        if isinstance(local_fusion, torch.Tensor):
            if local_fusion.dim() == 2:
                # [L, H] -> expect mask [L,H] or [1,L,H]
                if mask.dim() == 3 and mask.shape[0] >= 1:
                    mask_used = mask[0]
                elif mask.dim() == 2:
                    mask_used = mask
                else:
                    # fallback: try mean across first dim
                    mask_used = mask.mean(dim=0)
            elif local_fusion.dim() == 3:
                B = local_fusion.shape[0]
                # mask can be [B,L,H] or [N,L,H] where N>=B or [L,H]
                if mask.dim() == 3:
                    if mask.shape[0] == B:
                        mask_used = mask
                    elif mask.shape[0] > B:
                        mask_used = mask[:B]
                    elif mask.shape[0] == 1:
                        mask_used = mask.expand(B, -1, -1)
                    else:
                        # fallback to broadcasting along batch
                        mask_used = mask.mean(dim=0, keepdim=True).expand(B, -1, -1)
                elif mask.dim() == 2:
                    # [L,H] -> expand to [B,L,H]
                    mask_used = mask.unsqueeze(0).expand(B, -1, -1)
                else:
                    mask_used = mask
            else:
                raise ValueError("local_fusion must be 2D or 3D tensor")
        else:
            raise ValueError("local_fusion must be a torch.Tensor")

        if self.mask_type == 'multiplicative':
            masked_output = local_fusion * mask_used
            if return_mask:
                return masked_output, mask_used
            return masked_output

        elif self.mask_type == 'additive':
            # additive mask需要在外部应用到logits上
            if return_mask:
                return mask_used, mask_used
            return mask_used

        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")


# ---------------- Global Mask ----------------
class GlobalMask(nn.Module):
    def __init__(self, qmh5_file, hidden_dim, local_target_len, energy_threshold=1.0,
                 mask_type='multiplicative', mask_value=-1e9, device: torch.device = None):
        """
        全局掩码模块

        Args:
            qmh5_file: QM数据的h5文件路径
            hidden_dim: 隐藏层维度
            local_target_len: 局部特征长度
            energy_threshold: 能量阈值（<=threshold的保留，>threshold的mask掉）
            mask_type: 'multiplicative' 或 'additive'
                - 'multiplicative': 直接乘在特征上 (feature * mask)
                - 'additive': 加到logits上 (logits + mask)，用于attention机制
            mask_value: additive mask使用的值（默认-1e9）
        """
        super().__init__()
        self.energy_threshold = energy_threshold
        self.hidden_dim = hidden_dim
        self.local_target_len = local_target_len
        self.qmh5_file = qmh5_file
        self.mask_type = mask_type
        self.mask_value = mask_value

        # precompute global energy (on provided device when possible)
        _, self.energy_global = Energy(self.qmh5_file, local_target_len=self.local_target_len,
                                       hidden_dim=self.hidden_dim, device=device)  # [20,hidden_dim]
        # store mol ids and mapping for indexing by ID
        try:
            with h5py.File(self.qmh5_file, 'r') as fh:
                self.mol_ids = [str(k) for k in fh.keys()]
        except Exception:
            logging.exception("Failed to read mol ids from %s", str(self.qmh5_file))
            self.mol_ids = [str(i) for i in range(self.energy_global.size(0))] if isinstance(self.energy_global, torch.Tensor) else []
        try:
            self.id_to_idx = {mid: i for i, mid in enumerate(self.mol_ids)}
        except Exception:
            self.id_to_idx = {}
        self.linear1 = nn.Linear(self.hidden_dim, 5 * self.hidden_dim)  # [4,hidden_dim] -> [4,5*hidden_dim]
        self.linear2 = nn.Linear(5 * self.hidden_dim, self.hidden_dim)  # [4,5*hidden_dim] -> [4,hidden_dim]

    def get_bool_mask(self, batch_idx=None):
        """返回bool类型的mask，True表示保留，False表示屏蔽

        Args:
            batch_idx: None or list/tensor of indices or list of mol_id strings
        """
        if batch_idx is None:
            return self.energy_global <= self.energy_threshold

        # Safe, per-item indexing with validation to avoid CUDA device-side asserts
        try:
            N = self.energy_global.shape[0]
            default_row = self.energy_global[0]
            masks = []

            if isinstance(batch_idx, (list, tuple)) and len(batch_idx) > 0 and isinstance(batch_idx[0], str):
                idx_list = [self.id_to_idx.get(str(x), None) for x in batch_idx]
                bad = []
                for i, maybe in enumerate(idx_list):
                    if maybe is None or not (0 <= int(maybe) < N):
                        bad.append((i, batch_idx[i], maybe))
                        masks.append(default_row)
                    else:
                        masks.append(self.energy_global[int(maybe)])
                if bad:
                    logging.warning("GlobalMask: some mol ids not found or out-of-range: %s; using default row for those.", bad)
            else:
                # attempt to coerce into list of ints
                try:
                    idx_list = torch.as_tensor(batch_idx, dtype=torch.long).cpu().tolist()
                except Exception:
                    try:
                        single = int(batch_idx)
                        idx_list = [single]
                    except Exception:
                        logging.exception("GlobalMask: unable to coerce batch_idx=%s to indices; returning full mask.", str(batch_idx))
                        return self.energy_global <= self.energy_threshold

                if isinstance(idx_list, int):
                    idx_list = [idx_list]

                bad = []
                for i, v in enumerate(idx_list):
                    try:
                        vi = int(v)
                    except Exception:
                        bad.append((i, v))
                        masks.append(default_row)
                        continue
                    if not (0 <= vi < N):
                        bad.append((i, vi))
                        masks.append(default_row)
                    else:
                        masks.append(self.energy_global[vi])

                if bad:
                    logging.warning("GlobalMask: some provided indices are out-of-range or invalid: %s; using default row for those.", bad)

            if len(masks) == 0:
                logging.warning("GlobalMask: no valid indices found in batch_idx=%s; returning full mask.", str(batch_idx))
                return self.energy_global <= self.energy_threshold

            mask_tensor = torch.stack(masks, dim=0).to(self.energy_global.device)
            return (mask_tensor <= self.energy_threshold)

        except Exception:
            logging.exception("Failed to index energy_global with batch_idx=%s; fallback to full mask.", str(batch_idx))
            return self.energy_global <= self.energy_threshold

    def get_mask(self, batch_idx=None):
        """根据mask_type返回相应格式的mask，支持按 batch_idx 索引。"""
        bool_mask = self.get_bool_mask(batch_idx=batch_idx)

        if self.mask_type == 'additive':
            return bool_to_additive_mask(bool_mask, self.mask_value)
        elif self.mask_type == 'multiplicative':
            return bool_to_multiplicative_mask(bool_mask)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def forward(self, global_fusion, batch_idx=None, return_mask=False):
        """
        Args:
            global_fusion: [B, H] 全局融合特征
            return_mask: 是否返回mask本身

        Returns:
            masked_output: [B, H] 应用掩码后的特征
        """
        # Apply first linear projection
        out1 = self.linear1(global_fusion)  # [B, 5*hidden_dim]

        # determine batch size dynamically
        B = out1.shape[0]

        # reshape to [B, 5, hidden_dim] for elementwise operations with a per-token mask
        try:
            out1_reshaped = out1.view(B, 5, self.hidden_dim)
        except Exception:
            out1_reshaped = out1.reshape(B, 5, self.hidden_dim)

        # Prefer callers to provide batch_idx so we can select per-batch rows from the
        # precomputed global energy. If batch_idx is None, fallback to returning
        # the full dataset mask (older behaviour) and attempt best-effort alignment.
        mask = self.get_mask(batch_idx=batch_idx)  # expected shape: [B, hidden_dim] (or larger dataset-wise)

        # If mask does not match batch size, try to make it compatible by selecting or broadcasting
        if mask is not None and mask.dim() == 2:
            if mask.shape[0] == B:
                mask_used = mask
            elif mask.shape[0] == 1:
                # broadcast singleton
                mask_used = mask.expand(B, -1)
            elif mask.shape[0] > B:
                # best-effort: take first B rows (higher-level alignment should normally handle indexing)
                try:
                    mask_used = mask[:B]
                except Exception:
                    mask_used = mask.mean(dim=0, keepdim=True).expand(B, -1)
            else:
                # mask smaller than batch and not singleton: fallback to mean mask
                mask_used = mask.mean(dim=0, keepdim=True).expand(B, -1)
        else:
            mask_used = None

        if self.mask_type == 'multiplicative':
            if mask_used is not None:
                # mask_used: [B, hidden_dim] -> expand to [B, 1, hidden_dim] to multiply with out1_reshaped
                try:
                    masked_reshaped = out1_reshaped * mask_used.unsqueeze(1)
                except Exception:
                    # fallback: try broadcasting along last dim
                    masked_reshaped = out1_reshaped * mask_used.view(B, 1, -1)
            else:
                masked_reshaped = out1_reshaped
        elif self.mask_type == 'additive':
            # additive mode: do not modify features here, but return mask for external use
            masked_reshaped = out1_reshaped
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        # reshape back to [B, 5*hidden_dim] and apply linear2 to map to [B, hidden_dim]
        try:
            masked = masked_reshaped.view(B, 5 * self.hidden_dim)
        except Exception:
            masked = masked_reshaped.reshape(B, 5 * self.hidden_dim)

        masked = self.linear2(masked)

        if return_mask:
            return masked, mask
        return masked


# ---------------- 使用示例 ----------------
"""
使用方式1: Multiplicative Mask (直接乘在特征上)
适用场景：直接过滤特征，不满足条件的位置置0

local_mask = LocalMask(qmh5_file, hidden_dim=128, local_target_len=128, 
                       energy_threshold=1.0, mask_type='multiplicative')
global_mask = GlobalMask(qmh5_file, hidden_dim=128, local_target_len=128,
                         energy_threshold=1.0, mask_type='multiplicative')

# 直接应用mask
local_masked = local_mask(fusion_out["local_fusion"])   # [L, H] or [B, L, H]
global_masked = global_mask(fusion_out["global_fusion"]) # [B, H]


使用方式2: Additive Mask (加到logits上)
适用场景：用于attention机制，在softmax之前加到attention scores上

local_mask = LocalMask(qmh5_file, hidden_dim=128, local_target_len=128,
                       energy_threshold=1.0, mask_type='additive', mask_value=-1e9)

# 获取additive mask
additive_mask = local_mask(fusion_out["local_fusion"])  # [L, H] 或 [B, L, H]

# 在attention计算中使用
# attention_scores shape: [B, num_heads, L, L]
# 需要广播additive_mask到正确的shape
attention_scores = attention_scores + additive_mask.unsqueeze(0).unsqueeze(0)
attention_weights = torch.softmax(attention_scores, dim=-1)


使用方式3: 混合使用 - 在不确定性加权中应用
适用场景：将mask应用到不确定性权重矩阵上

# 1. 获取bool mask
bool_mask_local = local_mask.get_bool_mask()  # [L, H], True表示保留
bool_mask_global = global_mask.get_bool_mask()  # [20, H], True表示保留

# 2. 应用到不确定性矩阵
# weight_mat shape: [B, L, L]
weight_mat_masked = weight_mat * bool_mask_local.float()

# 3. 或转换为additive mask加到logits
additive_mask = bool_to_additive_mask(bool_mask_local, mask_value=-1e9)
logits = logits + additive_mask


完整示例代码:
---
qmh5_file = "path/to/qm.hdf5"

# 初始化mask (可选择不同类型)
local_mask_module = LocalMask(
    qmh5_file=qmh5_file,
    hidden_dim=128,
    local_target_len=128,
    energy_threshold=1.0,
    mask_type='multiplicative'  # 或 'additive'
)

global_mask_module = GlobalMask(
    qmh5_file=qmh5_file,
    hidden_dim=128,
    local_target_len=128,
    energy_threshold=1.0,
    mask_type='multiplicative'
)

# 应用mask
local_fused_masked = local_mask_module(fusion_out["local_fusion"])
global_fused_masked = global_mask_module(fusion_out["global_fusion"])

features_masked = {
    "local_fusion": local_fused_masked,
    "global_fusion": global_fused_masked
}
"""
