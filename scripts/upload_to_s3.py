#!/usr/bin/env python3
"""Upload dataset files from gdrift/data-sia/ to S3 with obfuscated (hashed) filenames.

By default, performs a dry-run showing planned renames. Pass --execute to upload.

Usage:
    python scripts/upload_to_s3.py              # dry-run
    python scripts/upload_to_s3.py --execute    # actually upload
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# Reuse the hash_name function from the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gdrift.datasetnames import hash_name  # noqa: E402

S3CMD_CONFIG = Path.home() / ".s3cfg-gadopt"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "gdrift" / "datasets.json"
DATA_SIA_DIR = Path(__file__).resolve().parent.parent / "gdrift" / "data-sia"


def load_manifest_names():
    """Return set of valid dataset names from the manifest."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    return {entry["name"] for entry in manifest["datasets"]}


def s3cmd(*args):
    """Run an s3cmd command with the gadopt config."""
    cmd = ["s3cmd", "-c", str(S3CMD_CONFIG)] + list(args)
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    elif result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform the upload (default is dry-run)",
    )
    parser.add_argument(
        "--no-delete", action="store_true",
        help="Skip deleting existing objects before uploading",
    )
    args = parser.parse_args()

    if not DATA_SIA_DIR.exists():
        print(f"ERROR: Source directory {DATA_SIA_DIR} does not exist.")
        print("Rename gdrift/data/ to gdrift/data-sia/ first:")
        print("  mv gdrift/data gdrift/data-sia")
        sys.exit(1)

    h5_files = sorted(DATA_SIA_DIR.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {DATA_SIA_DIR}")
        sys.exit(1)

    valid_names = load_manifest_names()

    # Build upload plan
    plan = []
    for filepath in h5_files:
        dataset_name = filepath.stem  # strip .h5
        if dataset_name not in valid_names:
            print(f"WARNING: Skipping {filepath.name} — not in datasets.json")
            continue
        hashed_filename = hash_name(dataset_name) + ".h5"
        plan.append((filepath, dataset_name, hashed_filename))

    # Print plan
    print(f"\n{'=' * 72}")
    print(f"Upload plan: {len(plan)} files from {DATA_SIA_DIR}")
    print(f"{'=' * 72}")
    for filepath, dataset_name, hashed_filename in plan:
        print(f"  {dataset_name:50s} -> {hashed_filename[:16]}...h5")
    print(f"{'=' * 72}\n")

    if not args.execute:
        print("DRY RUN — pass --execute to actually upload.")
        return

    # Delete existing objects
    if not args.no_delete:
        print("Deleting existing objects under s3://gadopt/g-drift/ ...")
        s3cmd("del", "--recursive", "s3://gadopt/g-drift/")
        print()

    # Upload each file
    s3_prefix = "s3://gadopt/g-drift/"
    for i, (filepath, dataset_name, hashed_filename) in enumerate(plan, 1):
        s3_dest = s3_prefix + hashed_filename
        print(f"[{i}/{len(plan)}] Uploading {dataset_name} -> {hashed_filename[:16]}...h5")
        result = s3cmd("put", "--acl-public", str(filepath), s3_dest)
        if result.returncode != 0:
            print(f"  FAILED to upload {dataset_name}")
            sys.exit(1)

    # Set public ACL on all uploaded files (belt and suspenders)
    print("\nSetting public-read ACL on all uploaded objects...")
    s3cmd("setacl", "--acl-public", "--recursive", s3_prefix)

    print(f"\nDone! Uploaded {len(plan)} files (public-read).")


if __name__ == "__main__":
    main()
