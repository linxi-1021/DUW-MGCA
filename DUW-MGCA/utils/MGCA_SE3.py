import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/DUW/data/components/'))
import h5py
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from data.components.datasets import MolDataset, ProtDataset
from data.components.transformQM import GNNTransformQM
from data.components.transformMD import GNNTransformMD
from data.qm_datamodule import QMDataModule
from data.md_datamodule import MDDataModule
from data.processing import preprocessing_db
from se3 import *
from QMExtractor import *
from MDSampler import *

hidden_dim = 64
num_layers = 4

# ===================== QM =====================
# 注意 edge_dim 要根据 batch_qm.edge_attr 的维度

model_qm = SE(
    node_dim=batch_qm.x.shape[1],    # use actual feature dim
    edge_dim=(batch_qm.edge_attr.size(-1) if hasattr(batch_qm, "edge_attr") and batch_qm.edge_attr is not None else 1),
    hidden_dim=hidden_dim,
    num_layers=num_layers
)

outputs_qm = model_qm(
    x=batch_qm.x,
    pos=batch_qm.pos,
    edge_index=(batch_qm.edge_index if hasattr(batch_qm, "edge_index") else None),
    edge_attr=(batch_qm.edge_attr if hasattr(batch_qm, "edge_attr") else None),
    batch=(batch_qm.batch if hasattr(batch_qm, "batch") else None)
)

print("QM local_se:", outputs_qm["local_se"].shape)   # [N_qm, hidden_dim]
print("QM global_se:", outputs_qm["global_se"].shape) # [num_graphs, hidden_dim]

# ===================== MD =====================
# 注意 MD 有时 edge_attr 为 None，可以设置 edge_dim = 1

model_md = SE(
    node_dim=batch_md.x.shape[1],
    edge_dim=1,
    hidden_dim=hidden_dim
)

outputs_md = model_md(
    x=batch_md.x,
    pos=batch_md.pos,
    edge_index=(batch_md.edge_index if hasattr(batch_md, "edge_index") else None),
    edge_attr=(batch_md.edge_attr if hasattr(batch_md, "edge_attr") else None),
    batch=(batch_md.batch if hasattr(batch_md, "batch") else None)
)

print("MD local_se:", outputs_md["local_se"].shape)
print("MD global_se:", outputs_md["global_se"].shape)