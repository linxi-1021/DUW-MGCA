'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import h5py

import torch
from torch.utils.data import Dataset


# =========================
# 工具函数
# =========================

def _normalize_id(x: str) -> str:
    """统一 ID 字符串格式：去空格 + 大写。"""
    return str(x).strip().upper()


def _load_metadata_target_map(csv_path: str, id_col: str, target_col: str) -> dict:
    """从 metadata.csv 读取 {规范化ID -> target} 映射。"""
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Metadata CSV 必须包含列: {id_col}, {target_col}")
    df = df[[id_col, target_col]].dropna()
    df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    return {row[id_col]: float(row[target_col]) for _, row in df.iterrows()}


# =========================
# MD / Prot Dataset
# =========================

class ProtDataset(Dataset):
    """加载 MD 数据集（蛋白坐标 + per-atom adaptability）。"""

    def __init__(self, md_data_file, idx_file, transform=None, post_transform=None):
        """
        Args:
            md_data_file (str): H5 文件路径
            idx_file (str): txt 文件路径，包含当前 split 的 pdb id 列表
            transform (callable): 将 dict 转为 PyG Data 的变换
            post_transform (callable): 数据增强（可选）
        """
        self.md_data_file = Path(md_data_file).absolute()
        with open(idx_file, 'r') as f:
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.md_data_file, 'r')
        self._transform = transform
        self._post_transform = post_transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.ids):
            raise IndexError(index)

        pid = self.ids[index]
        pitem = self.f[pid]

        # 只取蛋白部分的原子（cutoff 来自 molecules_begin_atom_index）
        cutoff = pitem["molecules_begin_atom_index"][:][-1]

        atoms_protein = pd.DataFrame({
            "x": pitem["atoms_coordinates_ref"][:][:cutoff, 0],
            "y": pitem["atoms_coordinates_ref"][:][:cutoff, 1],
            "z": pitem["atoms_coordinates_ref"][:][:cutoff, 2],
            "element": pitem["atoms_element"][:][:cutoff],
        })

        item = {
            "atoms_protein": atoms_protein,
            "scores": pitem["feature_atoms_adaptability"][:][:cutoff],
            "id": pid,
        }

        if self._transform:
            item = self._transform(item)
        if self._post_transform:
            item = self._post_transform(item)

        return item


# =========================
# QM / Mol Dataset
# =========================

class MolDataset(Dataset):
    """
    加载 QM 数据集。

    - 默认 label: [Electron_Affinity, Hardness]（2 维）
    - 若提供 metadata_csv，则使用 Binding_Affinity_kcal_mol（1 维标量）作为监督：
        . 过滤掉 metadata 中没有的 ID
        . labels = tensor([binding_affinity])
    - 不做任何标准化 / 反标准化，全部返回原始物理量。
    """

    def __init__(
        self,
        data_file,
        idx_file,
        target_norm_file=None,      # 为兼容旧接口，保留参数但不使用
        transform=None,
        isTrain=False,              # 同上，仅保留签名，内部不依赖
        post_transform=None,
        metadata_csv: str | None = None,
        metadata_id_col: str = 'PDBID',
        metadata_target_col: str = 'Binding_Affinity_kcal_mol',
        metadata_norm_mean: float | None = None,   # 兼容旧接口，忽略
        metadata_norm_std: float | None = None,    # 兼容旧接口，忽略
    ):
        """
        Args:
            data_file (str): QM H5 文件路径
            idx_file (str): txt 文件路径，包含当前 split 的 pdb id 列表
            target_norm_file: 兼容旧接口（不再使用）
            transform (callable): 将 dict 转为 PyG Data 的变换
            isTrain (bool): 兼容旧接口（不再使用）
            post_transform (callable): 数据增强（可选）
            metadata_csv (str): 若提供，则使用其中的 Binding_Affinity 作为监督目标
        """
        if isinstance(data_file, list):
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for h5")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        with open(idx_file, 'r') as f:
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.data_file, 'r')

        self._transform = transform
        self._post_transform = post_transform

        # 若提供 metadata，则构建 ID -> Binding_Affinity 映射，并过滤无标签样本
        self._target_map = None
        if metadata_csv:
            try:
                target_map = _load_metadata_target_map(
                    metadata_csv, metadata_id_col, metadata_target_col
                )
                ids_norm = [_normalize_id(i) for i in self.ids]
                before = len(self.ids)
                self.ids = [
                    orig for orig, nid in zip(self.ids, ids_norm)
                    if nid in target_map
                ]
                self._target_map = target_map
                logging.info(
                    f"[MolDataset] 使用 metadata 目标: {len(self.ids)}/{before} 条样本 "
                    f"(列: {metadata_id_col}, {metadata_target_col})"
                )
            except Exception as e:
                logging.warning(
                    f"[MolDataset] 载入 metadata 失败，回退到原始 QM labels: {e}"
                )

        # 兼容保留：默认的 QM 多任务标签名称
        self.target_dict = {
            "Electron_Affinity": 1,
            "Hardness": 3,
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.ids):
            raise IndexError(index)

        pid = self.ids[index]
        pitem = self.f[pid]

        # 原子坐标 + 元素
        prop = pitem["atom_properties"]["atom_properties_values"][:].astype(np.float32)
        atom_names = pitem["atom_properties/atom_names"][:]

        atoms = pd.DataFrame({
            "x": prop[:, 0],
            "y": prop[:, 1],
            "z": prop[:, 2],
            "element": np.array(
                [int(e.decode("utf-8")) for e in atom_names]
            ),
        })

        bonds = pitem["atom_properties/bonds"][:]

        # ========= 监督标签部分 =========
        if self._target_map is None:
            # 默认：从 QM H5 读取 Electron_Affinity / Hardness（原始值，不标准化）
            elec_aff = torch.tensor(
                pitem["mol_properties"]["Electron_Affinity"][()],
                dtype=torch.float32,
            )
            hardness = torch.tensor(
                pitem["mol_properties"]["Hardness"][()],
                dtype=torch.float32,
            )
            labels = torch.stack([elec_aff, hardness], dim=0)   # shape [2]
        else:
            # 使用 metadata 中的 Binding_Affinity_kcal_mol（原始值）
            nid = _normalize_id(pid)
            if nid not in self._target_map:
                raise KeyError(f"ID {pid} 缺少 metadata 目标")
            y_val = float(self._target_map[nid])
            labels = torch.tensor([y_val], dtype=torch.float32)  # shape [1]

        item = {
            "atoms": atoms,
            "labels": labels,
            "bonds": bonds,
            "id": pid,
        }

        if self._transform:
            item = self._transform(item)
        if self._post_transform:
            item = self._post_transform(item)

        return item
