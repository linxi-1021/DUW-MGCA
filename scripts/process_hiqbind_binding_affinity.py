"""
Script purpose: Process HiQBind metadata, normalize PDBIDs to uppercase,
compute Kd (M), LogKd, and binding free energy ΔG (kcal/mol),
optionally perform column-based or full-row deduplication, and output a cleaned affinity table.
"""
import pandas as pd
import numpy as np
import os

def process_hiqbind_metadata(input_csv, output_csv=None, temperature=298.15,
                             dedup=False, dedup_keys=None):
    """
    Process a HiQBind metadata CSV and compute binding affinity ΔG (kcal/mol) and LogKd.

    Args:
        input_csv (str): Path to input file (hiqbind_metadata.csv)
        output_csv (str): Path to output CSV
        temperature (float): Temperature in Kelvin (default: 298.15)
        dedup (bool): Whether to deduplicate the output rows
        dedup_keys (List[str] | None): Columns to use for deduplication; if None, full-row dedup is used
    Returns:
        pd.DataFrame: DataFrame containing computed columns
    """
    # 常数
    R = 1.987e-3  # kcal/mol/K
    UNIT_MAP = {
        'fM': 1e-15,
        'pM': 1e-12,
        'nM': 1e-9,
        'uM': 1e-6,
        'mM': 1e-3,
        'M': 1.0
    }

    print(f"📥 Loading metadata from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Check required columns
    required_cols = ["Binding Affinity Value", "Binding Affinity Unit", "PDBID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in {input_csv}")

    # Normalize PDBID to uppercase (strip whitespace)
    df["PDBID"] = df["PDBID"].astype(str).str.strip().str.upper()

    # Map units to multipliers
    df["unit_factor"] = df["Binding Affinity Unit"].map(UNIT_MAP)

    # Compute Kd (M)
    df["Kd_M"] = df["Binding Affinity Value"] * df["unit_factor"]

    # Filter invalid entries
    invalid_mask = df["Kd_M"].isna() | (df["Kd_M"] <= 0)
    invalid_count = invalid_mask.sum()
    df_valid = df.loc[~invalid_mask].copy()
    print(f"⚙️  Filtered out {invalid_count} invalid rows (missing/negative Kd)")

    # Compute LogKd
    df_valid["LogKd"] = np.log10(df_valid["Kd_M"].astype(np.float64))

    # Compute ΔG (kcal/mol)
    df_valid["Binding_Affinity_kcal_mol"] = R * temperature * np.log(df_valid["Kd_M"]).astype(np.float64)

    # More negative ΔG → stronger binding
    mean_val = df_valid["Binding_Affinity_kcal_mol"].mean()
    print(f"📊 Mean Binding Free Energy: {mean_val:.3f} kcal/mol")

    # Mark valid samples
    df_valid["Valid"] = True

    # Select output columns
    keep_cols = [
        "PDBID", "Binding Affinity Measurement", "Binding Affinity Value",
        "Binding Affinity Unit", "Kd_M", "LogKd",
        "Binding_Affinity_kcal_mol", "Ligand Name", "Ligand SMILES",

    ]

    df_out = df_valid[[c for c in keep_cols if c in df_valid.columns]]

    # Deduplication logic
    before_rows = len(df_out)
    if dedup:
        if dedup_keys:
            # Deduplicate based on specified columns only
            existing_keys = [k for k in dedup_keys if k in df_out.columns]
            if existing_keys:
                df_out = df_out.drop_duplicates(subset=existing_keys, keep='first')
            else:
                print(f"⚠️ Specified dedup columns {dedup_keys} not present in output; performing full-row dedup")
                df_out = df_out.drop_duplicates(keep='first')
        else:
            # Full-row deduplication
            df_out = df_out.drop_duplicates(keep='first')
    after_rows = len(df_out)
    if dedup:
        print(f"🔁 Dedup: {before_rows} -> {after_rows} rows (reduced {before_rows - after_rows})")

    # Save output file
    if output_csv is None:
        output_csv = os.path.splitext(input_csv)[0] + "_clean_binding_affinity.csv"

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(output_csv, index=False)
    print(f"✅ Cleaned file saved to: {output_csv}")
    print(f"🧾 Total valid entries: {len(df_out)}")

    return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process HiQBind metadata to compute ΔG (kcal/mol).")
    parser.add_argument("--input", type=str, default="../hiqbind_metadata.csv", help="Path to hiqbind_metadata.csv")
    parser.add_argument("--output", type=str, default="../data/metadata.csv", help="Path to output CSV")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in Kelvin (default=298.15K)")
    parser.add_argument("--dedup", action='store_true', default=True, help="Deduplicate output CSV")
    parser.add_argument("--dedup_keys", type=str, default="PDBID", help="Comma-separated column names to deduplicate on; defaults to full-row dedup if empty")

    args = parser.parse_args()

    dedup_keys = None
    if args.dedup_keys:
        dedup_keys = [x.strip() for x in args.dedup_keys.split(',') if x.strip()] or None
    df = process_hiqbind_metadata(
        args.input,
        args.output,
        args.temperature,
        dedup=args.dedup,
        dedup_keys=dedup_keys,
    )
