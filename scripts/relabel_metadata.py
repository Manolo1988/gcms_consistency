#!/usr/bin/env python3
"""Safely relabel selected samples in prepared metadata.csv.

This edits label columns only. Tensor files are not moved; existing tensor_path
values remain valid and traceable to the original prepared folder.
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = [
    "code",
    "fine_code",
    "coarse_code",
    "product_code",
    "product_fine",
    "product_coarse",
]


def _parse_sample_ids(args):
    ids = []
    if args.sample_ids:
        ids.extend([x.strip() for x in args.sample_ids.split(",") if x.strip()])
    if args.sample_file:
        table = pd.read_csv(args.sample_file)
        if "sample_id" not in table.columns:
            raise SystemExit("--sample_file must contain a sample_id column")
        ids.extend([str(x).strip() for x in table["sample_id"].tolist() if str(x).strip()])
    return list(dict.fromkeys(ids))


def main():
    parser = argparse.ArgumentParser(description="Relabel samples in metadata.csv")
    parser.add_argument("--metadata", default="new_prepared_data/metadata.csv")
    parser.add_argument("--sample_ids", default="", help="Comma-separated sample ids")
    parser.add_argument("--sample_file", default="", help="CSV with sample_id column")
    parser.add_argument("--new_label", required=True, help="New product label, e.g. YJD")
    parser.add_argument("--new_spec_name", default="", help="Override spec_name")
    parser.add_argument("--output", default="", help="Output CSV; default edits metadata in place")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise SystemExit(f"metadata not found: {metadata_path}")

    sample_ids = _parse_sample_ids(args)
    if not sample_ids:
        raise SystemExit("No sample ids provided")

    df = pd.read_csv(metadata_path)
    if "sample_id" not in df.columns:
        raise SystemExit("metadata.csv has no sample_id column")

    new_label = str(args.new_label).strip()
    mask = df["sample_id"].astype(str).isin(sample_ids)
    missing = sorted(set(sample_ids) - set(df.loc[mask, "sample_id"].astype(str)))
    if missing:
        print("[WARN] sample ids not found:")
        for sid in missing:
            print(f"  {sid}")

    if not mask.any():
        raise SystemExit("No matching rows to relabel")

    spec_name = str(args.new_spec_name).strip()
    if not spec_name and "spec_name" in df.columns:
        candidates = df.loc[df["product_fine"].astype(str) == new_label, "spec_name"]
        candidates = candidates.dropna().astype(str)
        if not candidates.empty:
            spec_name = candidates.mode().iloc[0]

    before_cols = ["sample_id", "batch_name", "d_name", "spec_name"] + [
        c for c in LABEL_COLUMNS if c in df.columns
    ]
    before_cols = [c for c in before_cols if c in df.columns]
    print("[Relabel preview]")
    print(df.loc[mask, before_cols].to_string(index=False))

    for col in LABEL_COLUMNS:
        if col in df.columns:
            df.loc[mask, col] = new_label
    if spec_name and "spec_name" in df.columns:
        df.loc[mask, "spec_name"] = spec_name

    after_cols = before_cols
    print("\n[After]")
    print(df.loc[mask, after_cols].to_string(index=False))

    if args.dry_run:
        print("\nDry run only; metadata was not written.")
        return

    output_path = Path(args.output) if args.output else metadata_path
    if output_path == metadata_path and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = metadata_path.with_suffix(metadata_path.suffix + f".bak_{stamp}")
        shutil.copy2(metadata_path, backup_path)
        print(f"\nBackup written: {backup_path}")

    df.to_csv(output_path, index=False)
    print(f"Relabeled {int(mask.sum())} rows -> {new_label}")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
