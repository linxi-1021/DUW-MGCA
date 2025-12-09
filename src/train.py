import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/data/components/'))

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
#
class Args:
    # input file
    datasetIn = "../data/MD/h5_files/MD.hdf5"
    # Feature that should be stripped, e.g. atoms_element or atoms_type
    strip_feature = "atoms_element"
    # Value to strip, e.g. if strip_freature= atoms_element; 1 for H.
    strip_value = -1
    # Start index of structures
    begin = 9000
    # End index of structures
    end = -1
    # We calculate the adaptability for each atom.
    # Default behaviour will also strip H atoms, if no stripping should be perfomed set strip_value to -1.
    Adaptability = True
    # If set to True this will create a new feature that combines one entry for each protein AA but all ligand entries;
    # e.g. for only ca set strip_feature = atoms_type and strip_value = 14
    Pres_Lat = False
    # We strip the complex by given distance (in Angstrom) from COG of molecule,
    # use e.g. 15.0. If default value is given (0.0) no pocket stripping will be applied.
    Pocket = 0.0
    # output file name and location
    # datasetOut = "../data/MD/h5_files/md_out1.hdf5"
    datasetOut = "G:/md_out2.hdf5"



args = Args()

preprocessing_db.main(args)
#
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
# train_dataset = ProtDataset(mdh5_file, idx_file=train_idx, transform=GNNTransformMD(), post_transform=T.RandomTranslate(0.05))
#
# train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0)
# for idx, val in enumerate(train_loader):
#     print(val)
#     break
#
# mddata = MDDataModule(files_root, h5file=mdh5_file, train=train_idx, val=val_idx, test=test_idx, batch_size=16,num_workers=0)
# mddata.setup()
# train_loader = mddata.train_dataloader()
#
# for idx, val in enumerate(train_loader):
#     print(val)
#     break