import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/DUW/data/components/'))

from DUW.data.md_datamodule import MDDataModule

batch_size = 4
num_workers = 0

files_root =  ""
mdh5_file = '../data/MD/h5_files/tiny_md_out.hdf5'
train_idx = "../data/MD/splits/train_tinyMD.txt"
val_idx = "../data/MD/splits/val_tinyMD.txt"
test_idx = "../data/MD/splits/test_tinyMD.txt"

mddata = MDDataModule(files_root, h5file=mdh5_file, train=train_idx, val=val_idx, test=test_idx, batch_size=batch_size,num_workers=num_workers)
mddata.setup()
md_loader = mddata.train_dataloader()
batch_md = next(iter(md_loader))

# print(batch_md)

# batch_md = next(iter(md_loader))



# files_root =  ""
#
# mdh5_file = '../data/MD/h5_files/tiny_md_out.hdf5'
#
# train_idx = "../data/MD/splits/train_tinyMD.txt"
# val_idx = "../data/MD/splits/val_tinyMD.txt"
# test_idx = "../data/MD/splits/test_tinyMD.txt"
#
# md_H5File = h5py.File(mdh5_file)
#
#
#
# mddata = MDDataModule(files_root, h5file=mdh5_file, train=train_idx, val=val_idx, test=test_idx, batch_size=8,num_workers=0)
# mddata.setup()
# train_loader = mddata.train_dataloader()
#
# # for idx, val in enumerate(train_loader):
# #     print(val)
# #     break
# # 你打印出来的 DataBatch
# batch = next(iter(train_loader))  # 直接拿 MISATO 的 QM/MD DataLoader
#
# model = EGNN(in_dim=11, edge_dim=1, hidden_dim=64, num_layers=4,node_level=True)
#
# node_feat_qm, node_pos_qm = model(batch)
# print("预测输出:", node_feat_qm.shape,node_pos_qm.shape)  # [batch_size, 1]