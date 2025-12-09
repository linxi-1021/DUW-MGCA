"""
Script purpose: Extract molecule IDs that are common across QM and MD HDF5 files and metadata.csv,
then randomly split them into train/val/test sets and output corresponding ID lists and statistics.
"""
import h5py
import pandas as pd
import random
import os

def get_common_ids(qm_h5, md_h5, metadata_csv, id_col="PDBID"):
    """Extract molecule IDs common to QM, MD, and metadata."""
    print(f"📂 Loading QM file: {qm_h5}")
    with h5py.File(qm_h5, "r") as f_qm:
        qm_ids = set(f_qm.keys())
    print(f"✅ QM samples: {len(qm_ids)}")

    print(f"📂 Loading MD file: {md_h5}")
    with h5py.File(md_h5, "r") as f_md:
        md_ids = set(f_md.keys())
    print(f"✅ MD samples: {len(md_ids)}")

    # QM–MD intersection
    common_h5 = qm_ids.intersection(md_ids)
    print(f"🔍 Common QM–MD samples: {len(common_h5)}")

    # Load metadata
    meta_df = pd.read_csv(metadata_csv)
    if id_col not in meta_df.columns:
        raise ValueError(f"metadata.csv does not contain column '{id_col}'")

    meta_df[id_col] = meta_df[id_col].astype(str).str.strip().str.upper()
    meta_ids = set(meta_df[id_col])
    print(f"✅ Metadata entries (uppercased): {len(meta_ids)}")

    # Intersection across QM, MD, and metadata
    common_all = sorted(list(common_h5.intersection(meta_ids)))
    print(f"🎯 Common QM∩MD∩Metadata: {len(common_all)}")
    return common_all


def split_dataset(ids, output_dir="splits", n_total=100, seed=42):
    """Randomly sample n_total entries and split into train/val/test."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    if len(ids) < n_total:
        raise ValueError(f"❌ Insufficient samples: {len(ids)} available, but {n_total} requested")

    selected = random.sample(ids, n_total)
    random.shuffle(selected)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ids = selected[:n_train]
    val_ids = selected[n_train:n_train + n_val]
    test_ids = selected[-n_test:]

    def save_list(path, data):
        with open(path, "w") as f:
            f.writelines([i + "\n" for i in data])

    save_list(os.path.join(output_dir, "train_ids.txt"), train_ids)
    save_list(os.path.join(output_dir, "val_ids.txt"), val_ids)
    save_list(os.path.join(output_dir, "test_ids.txt"), test_ids)

    # Summary
    print("\n✅ Split completed:")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print(f"💾 Files saved to: {output_dir}/")
    print(f"  - train_ids.txt")
    print(f"  - val_ids.txt")
    print(f"  - test_ids.txt")

    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select 100 common QM–MD–Metadata molecules and split into train/val/test sets")
    parser.add_argument("--qm_h5", type=str, default="../data/QM/h5_files/QM.hdf5", help="Path to QM HDF5 file (e.g. more_qm.hdf5)")
    parser.add_argument("--md_h5", type=str, default="../data/MD/h5_files/MD.hdf5", help="Path to MD HDF5 file (e.g. more_md_out.hdf5)")
    parser.add_argument("--metadata_csv", type=str, default="../data/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--output_dir", type=str, default="splits_100", help="Output directory for split txt files")
    parser.add_argument("--n_total", type=int, default=100, help="Number of molecules to sample (default=100)")
    parser.add_argument("--id_col", type=str, default="PDBID", help="Column name in metadata (default=PDBID)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    common_ids = get_common_ids(args.qm_h5, args.md_h5, args.metadata_csv, args.id_col)
    split_dataset(common_ids, args.output_dir, args.n_total, args.seed)
