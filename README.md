# DUW-MGCA
## 1. Download the MISATO dataset  

Original repository: <https://github.com/t7morgen/misato-dataset>  

Download the full MISATO dataset (make sure you have enough disk space, MD is about 133 GB):  

- **MD (~133 GB)**  
  - Direct download:  
    <https://zenodo.org/records/7711953/files/MD.hdf5?download=1>  
  - Or via command line:  
    ```bash
    wget -O data/MD/h5_files/MD.hdf5       https://zenodo.org/record/7711953/files/MD.hdf5
    ```

- **QM (~0.3 GB)**  
  - Direct download:  
    <https://zenodo.org/records/7711953/files/QM.hdf5?download=1>  
  - Or via command line:  
    ```bash
    wget -O data/QM/h5_files/QM.hdf5       https://zenodo.org/record/7711953/files/QM.hdf5
    ```

- **Original MD splits (optional, for reproducing the authors’ split)**  
  - `test_MD.txt` (~8.1 KB): <https://zenodo.org/records/7711953/files/test_MD.txt?download=1>  
  - `train_MD.txt` (~68.8 KB): <https://zenodo.org/records/7711953/files/train_MD.txt?download=1>  
  - `val_MD.txt` (~8.0 KB): <https://zenodo.org/records/7711953/files/val_MD.txt?download=1>  


## 2. Explore the MISATO data structure (`src/getting_started.ipynb`)

Use the notebook `src/getting_started.ipynb` to quickly explore the dataset and its PyTorch interfaces. It demonstrates:

1. The HDF5 hierarchy and how to access per-molecule properties and trajectories.  
2. How to instantiate QM / MD `Dataset` and `DataLoader` objects.  
3. How to iterate over `DataLoader`s and inspect batched samples, as preparation for model training.  


## 3. Preprocess MD data: pocket cropping and HDF5 rewriting  

Use `run_md_preprocess.py` (adjust the path according to your project layout) to crop, filter, and rewrite the original `MD.hdf5`, producing a new HDF5 tailored for graph neural network training:

```bash
python run_md_preprocess.py   --datasetIn MD.hdf5   --datasetOut new_MD.hdf5   --begin 0   --end 5000   --Pocket 8.0
```

Argument description:

- `--datasetIn`: path to the original MD HDF5 file.  
- `--datasetOut`: output path or filename for the preprocessed MD HDF5 file.  
- `--begin` / `--end`: index range of molecules to include (typically a subset for experiments).  
- `--Pocket`: pocket cutoff radius (Å) for cropping protein atoms around the ligand, e.g., `8.0`.  


## 4. Download HiQBind metadata  

Original repository: <https://github.com/THGLab/HiQBind>  

Download the binding affinity metadata:

- `hiqbind_metadata.csv`:  
  <https://github.com/THGLab/HiQBind/blob/main/figshare/hiqbind_metadata.csv>  

It is recommended to place this file under the project root or `data/` directory for easier scripting.  


## 5. Process HiQBind binding affinity data  

Use `scripts/process_hiqbind_binding_affinity.py` (here referred to as `process_hiqbind_metadata.py`) to clean and normalize the original `hiqbind_metadata.csv` into a modeling-ready binding affinity table. The script typically:

- Normalizes the `PDBID` format.  
- Converts different affinity units to molar concentration.  
- Computes `LogKd` and `ΔG` (kcal/mol).  
- Optionally removes duplicates.  
- Outputs a clean `metadata.csv`.  

Example command:

```bash
python process_hiqbind_metadata.py   --input ../hiqbind_metadata.csv   --output ../data/metadata.csv   --temperature 298.15   --dedup   --dedup_keys PDBID
```

Arguments:

- `--temperature`: temperature (K) used for ΔG computation, typically `298.15`.  
- `--dedup`: whether to deduplicate rows in the output.  
- `--dedup_keys`: column(s) used as keys to define duplicates, e.g., `PDBID`.  


## 6. Extract molecule IDs common to MD, QM, and metadata  

From the QM and MD HDF5 files, obtain the intersection of molecule IDs, then cross-filter with `metadata.csv` to keep only complexes that exist in all three sources. Save the final ID list and statistics to disk:

```bash
python extract_common_ids.py   --qm_h5 ../data/QM/h5_files/QM.hdf5   --md_h5 ../data/MD/h5_files/MD.hdf5   --metadata_csv ../data/metadata.csv   --output_dir ../data   --output common_ids.txt
```

This script will generate:

- `common_ids.txt`: list of PDBIDs present in QM, MD, and metadata.  
- Additional statistics or log files (depending on implementation) to inspect coverage.  


## 7. Split common IDs into new train / val / test sets  

Based on the shared IDs from QM, MD, and `metadata.csv`, randomly sample `n_total` complexes and split them into train / val / test sets with an 8:1:1 ratio. The IDs for each subset are stored as text files:

```bash
python split_common_ids.py   --qm_h5 ../data/QM/h5_files/QM.hdf5   --md_h5 ../data/MD/h5_files/MD.hdf5   --metadata_csv ../data/metadata.csv   --output_dir splits_100   --n_total 10000   --id_col PDBID   --seed 42
```

Arguments:

- `--output_dir`: directory where the split files are written, e.g., `splits_100`.  
- `--n_total`: total number of samples to participate in splitting (drawn from the common IDs).  
- `--id_col`: column name used as the unique identifier, usually `PDBID`.  
- `--seed`: random seed for reproducible splits.  

The output directory typically contains:

- `train_ids.txt`  
- `val_ids.txt`  
- `test_ids.txt`  


## 8. Example project structure  

```text
├── data                      <- Project data
│   ├── MD
│   │   ├── h5_files          <- Raw and preprocessed MD HDF5 files
│   │   └── splits            <- Train / val / test ID splits for MD
│   ├── QM
│   │   ├── h5_files          <- Raw and preprocessed QM HDF5 files
│   │   └── splits            <- Train / val / test ID splits for QM
│   ├── metadata.csv          <- Cleaned binding affinity table (from HiQBind)
│   └── common_ids.txt        <- Intersection IDs of QM / MD / metadata
│
├── src                       <- Source code
│   ├── data
│   │   ├── components        <- Dataset classes and data transforms
│   │   ├── md_datamodule.py  <- MD Lightning DataModule
│   │   ├── qm_datamodule.py  <- QM Lightning DataModule
│   │   └── processing        <- Preprocessing, inference, and conversion scripts
│   │       ├── run_md_preprocess.py
│   │       ├── extract_common_ids.py
│   │       ├── split_common_ids.py
│   │       └── process_hiqbind_metadata.py
│   ├── getting_started.ipynb <- Notebook: how to load and interact with the data
│   └── train.py              <- (Optional) model training entry point
│
└── README.md
