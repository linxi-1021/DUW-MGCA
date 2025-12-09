import torch
from UE_QM import *
from UE_MD import *


qmh5_file = '../../data/QM/h5_files/tiny_qm.hdf5'
mdh5_file = '../../data/MD/h5_files/tiny_md.hdf5'

qm_results = QMUncertaintyEstimator(qmh5_file).forward()
md_results = MDUncertaintyEstimator(mdh5_file).forward()

qm_mat,qm_global = qm_results["uncertainty_matrix"],qm_results["global_uncertainty_vector"]
md_mat,md_global = md_results["uncertainty_matrix"],md_results["global_uncertainty_vector"]

weight_mat = torch.einsum("bi,bj->bij", qm_results["uncertainty_matrix"], md_results["uncertainty_matrix"])
global_weight_mat = torch.einsum("b,b->b", qm_results["global_uncertainty_vector"], md_results["global_uncertainty_vector"])


print("QM 不确定性矩阵:\n", qm_mat,qm_mat.shape)
print("MD 不确定性矩阵:\n", md_mat,md_mat.shape)
print("Token-level 权重矩阵:\n", weight_mat,weight_mat.shape)
print("QM 全局不确定性:\n", qm_global,qm_global.shape)
print("MD 全局不确定性:\n", md_global,md_global.shape)
print("Molecule-level 权重矩阵:\n", global_weight_mat,global_weight_mat.shape)
