import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/DUW/data/components/'))

from DUW.data.qm_datamodule import QMDataModule

batch_size = 4
num_workers = 0

files_root =  "../data/QM"
qmh5file = "h5_files/tiny_qm.hdf5"
tr = "splits/train_tinyQM.txt"
v = "splits/val_tinyQM.txt"
te = "splits/test_tinyQM.txt"

qmdata = QMDataModule(files_root, h5file=qmh5file, train=tr, val=v, test=te,batch_size=batch_size, num_workers=num_workers)
qmdata.setup()
qm_loader = qmdata.train_dataloader()
batch_qm = next(iter(qm_loader))

# print(batch_qm)
# batch_qm = next(iter(qm_loader))


# files_root =  "../data/QM"
#
# qmh5file = "h5_files/tiny_qm.hdf5"
#
# tr = "splits/train_tinyQM.txt"
# v = "splits/val_tinyQM.txt"
# te = "splits/test_tinyQM.txt"
#
# qmdata = QMDataModule(files_root, h5file=qmh5file, train=tr, val=v, test=te,batch_size=8, num_workers=0)
# qmdata.setup()
# train_loader = qmdata.train_dataloader()
#
# # for idx, val in enumerate(train_loader):
# #     print(val)
# #     break
#
#
# # 你打印出来的 DataBatch
# batch = next(iter(train_loader))  # 直接拿 MISATO 的 QM/MD DataLoader
#
# model = EGNN(in_dim=25, edge_dim=1, hidden_dim=64, num_layers=4,node_level=True)
#
# node_feat_qm, node_pos_qm = model(batch)
# print("预测输出:", node_feat_qm.shape,node_pos_qm.shape)  # [batch_size, 1]