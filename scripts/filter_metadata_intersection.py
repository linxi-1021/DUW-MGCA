"""
Script purpose: Filter metadata.csv to keep only molecules present in both QM and MD HDF5 top-level keys.
Outputs intersection files and supports automatic ID column handling.
"""
import h5py
import pandas as pd
import os


def extract_common_ids_with_metadata(qm_h5, md_h5, metadata_csv, output_dir="outputs", id_col="PDBID"):
    """
    Extract molecule IDs that are common to QM and MD HDF5 files and present in metadata.csv
    (PDBIDs are normalized to uppercase).

    Args:
        qm_h5 (str): Path to QM HDF5 file
        md_h5 (str): Path to MD HDF5 file
        metadata_csv (str): Path to metadata.csv file
        output_dir (str): Output directory
        id_col (str): Column name in metadata used for matching (default: PDBID)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Loading QM file: {qm_h5}")
    with h5py.File(qm_h5, "r") as f_qm:
        qm_ids = set(f_qm.keys())
    print(f"✅ QM samples: {len(qm_ids)}")

    print(f"📂 Loading MD file: {md_h5}")
    with h5py.File(md_h5, "r") as f_md:
        md_ids = set(f_md.keys())
    print(f"✅ MD samples: {len(md_ids)}")

    # Compute intersection QM ∩ MD
    common_h5_ids = sorted(list(qm_ids.intersection(md_ids)))
    print(f"🔍 Common QM–MD samples: {len(common_h5_ids)}")

    # Load metadata.csv
    print(f"📑 Loading metadata: {metadata_csv}")
    meta_df = pd.read_csv(metadata_csv)

    if id_col not in meta_df.columns:
        raise ValueError(f"❌ metadata.csv does not contain column '{id_col}'")

    # --- ✅ Normalize to uppercase and strip ---
    meta_df[id_col] = meta_df[id_col].astype(str).str.strip().str.upper()
    metadata_ids = set(meta_df[id_col])
    print(f"✅ Metadata entries after uppercasing: {len(metadata_ids)}")

    # --- Compute intersection across QM, MD, and metadata ---
    common_valid_ids = sorted(list(set(common_h5_ids).intersection(metadata_ids)))
    missing_in_meta = sorted(list(set(common_h5_ids) - metadata_ids))

    print(f"🎯 Final valid common samples (in QM, MD, and metadata): {len(common_valid_ids)}")
    print(f"⚠️ Missing in metadata: {len(missing_in_meta)}")

    # --- Save output files ---
    common_path = os.path.join(output_dir, "common_ids.txt")
    with open(common_path, "w") as f:
        f.writelines([i + "\n" for i in common_valid_ids])

    missing_path = os.path.join(output_dir, "missing_in_metadata.txt")
    with open(missing_path, "w") as f:
        f.writelines([i + "\n" for i in missing_in_meta])

    # --- Write summary ---
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== DUW-MGCA Data Summary ===\n")
        f.write(f"QM total: {len(qm_ids)}\n")
        f.write(f"MD total: {len(md_ids)}\n")
        f.write(f"QM∩MD total: {len(common_h5_ids)}\n")
        f.write(f"Metadata total (uppercased): {len(metadata_ids)}\n")
        f.write(f"Final valid (QM∩MD∩Metadata): {len(common_valid_ids)}\n")
        f.write(f"Missing in metadata: {len(missing_in_meta)}\n")

    print(f"\n💾 Saved results to: {output_dir}")
    print(f"  - Common IDs: {common_path}")
    print(f"  - Missing IDs: {missing_path}")
    print(f"  - Summary: {summary_path}")

    return common_valid_ids, missing_in_meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract common molecule IDs from QM and MD HDF5 files.")
    parser.add_argument("--qm_h5", type=str, default="../data/QM/h5_files/QM.hdf5", help="Path to QM HDF5 file (e.g. more_qm.hdf5)")
    parser.add_argument("--md_h5", type=str, default="../data/MD/h5_files/MD.hdf5", help="Path to MD HDF5 file (e.g. more_md_out.hdf5)")
    parser.add_argument("--output", type=str, default="common_ids.txt", help="Output text file path")
    parser.add_argument("--metadata_csv", type=str, default="../data/metadata.txt", help="Path to metadata.csv file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--id_col", type=str, default="PDBID", help="Column name used as ID in metadata")

    args = parser.parse_args()

    extract_common_ids_with_metadata(args.qm_h5, args.md_h5, args.metadata_csv, args.output_dir, args.id_col)
