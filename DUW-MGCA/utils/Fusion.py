import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/DUW/data/components/'))
from DUW.utils.MGCA_SE3 import *


hidden_dim = outputs_qm["local_se"].size(1)  # H
fusion_hidden = 128
local_len = 128

fusion_model = Fusion(hidden_dim=hidden_dim, fusion_hidden_dim=fusion_hidden, local_target_len=local_len)
fusion_model.eval()

fusion_out = fusion_model(outputs_qm, outputs_md)

print("local_fusion shape:", fusion_out["local_fusion"].shape)   # [local_len, fusion_hidden]
print("global_fusion shape:", fusion_out["global_fusion"].shape) # [B, fusion_hidden]
