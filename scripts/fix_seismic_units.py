#!/usr/bin/env python
"""Fix seismic model HDF5 files that store absolute velocities in km/s.

Some seismic tomography HDF5 files store absolute velocity fields (vs, vp,
vsh, vsv, vpv, vph) in km/s instead of the expected m/s. This script:

1. Identifies affected fields in each model's HDF5 file.
2. Multiplies those fields by 1000 to convert km/s -> m/s.
3. Saves the updated HDF5 file.
4. Recomputes SHA256 hashes.
5. Updates datasets.json with the new hashes.

Usage:
    python scripts/fix_seismic_units.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Add gdrift to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gdrift.datasetnames import hash_name
from gdrift.io import path_to_dataset, file_hash

# Absolute velocity field names (perturbation fields like dvs/dvp are dimensionless)
ABS_VELOCITY_FIELDS = {"vs", "vp", "vsh", "vsv", "vpv", "vph"}

# Threshold: if max absolute velocity < 100, it's in km/s
KMS_THRESHOLD = 100.0


def find_affected_fields(h5path):
    """Return list of fields that are in km/s (need *1000)."""
    affected = []
    with h5py.File(h5path, "r") as f:
        for key in f.keys():
            if key not in ABS_VELOCITY_FIELDS:
                continue
            data = f[key][()]
            if data.max() < KMS_THRESHOLD:
                affected.append(key)
    return affected


def fix_file(h5path, fields, dry_run=False):
    """Multiply specified fields by 1000 in-place."""
    if dry_run:
        print(f"  [DRY RUN] Would fix fields: {fields}")
        return
    with h5py.File(h5path, "r+") as f:
        for field in fields:
            data = f[field][()]
            f[field][...] = data * 1000.0
            new_max = f[field][()].max()
            print(f"  Fixed {field}: max was {data.max():.4f} km/s, "
                  f"now {new_max:.1f} m/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be changed without modifying files")
    args = parser.parse_args()

    manifest_path = REPO_ROOT / "gdrift" / "datasets.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    seismic_datasets = [
        d for d in manifest["datasets"]
        if d["name"].startswith("3d_seismic_")
    ]

    fixed_models = []
    new_hashes = {}

    for ds in sorted(seismic_datasets, key=lambda d: d["name"]):
        name = ds["name"]
        h5name = hash_name(name) + ".h5"
        h5path = path_to_dataset(h5name)

        if not h5path.exists():
            continue

        affected = find_affected_fields(h5path)
        if not affected:
            continue

        print(f"\n{name}:")
        print(f"  Fields in km/s: {affected}")
        fix_file(h5path, affected, dry_run=args.dry_run)

        if not args.dry_run:
            new_hash = file_hash(h5path)
            new_hashes[name] = new_hash
            print(f"  New hash: {new_hash}")
            fixed_models.append(name)

    if not args.dry_run and new_hashes:
        # Update manifest
        for ds in manifest["datasets"]:
            if ds["name"] in new_hashes:
                old_hash = ds["sha256"]
                ds["sha256"] = new_hashes[ds["name"]]
                print(f"\nUpdated {ds['name']}:")
                print(f"  Old: {old_hash}")
                print(f"  New: {ds['sha256']}")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"\nUpdated {manifest_path} with {len(new_hashes)} new hashes.")

    print(f"\n{'Would fix' if args.dry_run else 'Fixed'} "
          f"{len(fixed_models if not args.dry_run else affected)} models.")

    if not args.dry_run and fixed_models:
        print("\nIMPORTANT: You need to re-upload the fixed HDF5 files to "
              "Digital Ocean Spaces for CI and other users to get the corrected data.")


if __name__ == "__main__":
    main()
