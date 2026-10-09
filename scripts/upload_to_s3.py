#!/usr/bin/env python3
"""Upload dataset files from gdrift/data-sia/ to S3 under their content names.

By default, performs a dry-run showing the planned uploads. Pass --execute to upload.

Each file is uploaded as `<sha256 of its content>.h5`, the `filename` that
datasets.json gives for the dataset. The script never deletes or overwrites a
file on the server: files that are already there are skipped, because older
gdrift releases can point to them. Update datasets.json (sha256 and filename)
before uploading; a file whose sha256 does not match its manifest entry is not
uploaded. See docs/dataset-releases.md.

Usage:
    python scripts/upload_to_s3.py              # dry-run
    python scripts/upload_to_s3.py --execute    # actually upload
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the hashing function from the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gdrift.io import file_hash  # noqa: E402

S3CMD_CONFIG = Path.home() / ".s3cfg-gadopt"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "gdrift" / "datasets.json"
DATA_SIA_DIR = Path(__file__).resolve().parent.parent / "gdrift" / "data-sia"


def load_manifest():
    """Return the parsed datasets.json manifest."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def object_exists(manifest, filename):
    """Return True if an object with this file name exists on the server (unsigned HEAD)."""
    s3 = manifest["s3"]
    url = f"{s3['endpoint_url'].rstrip('/')}/{s3['bucket']}/{s3['prefix']}{filename}"
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        return True
    except urllib.error.HTTPError as e:
        # The server answers 404 for a missing key. Any other error (403,
        # throttling) is unexpected, and an upload could then overwrite an
        # existing object, so stop.
        if e.code == 404:
            return False
        raise


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
    args = parser.parse_args()

    if not DATA_SIA_DIR.exists():
        print(f"ERROR: Source directory {DATA_SIA_DIR} does not exist.")
        print("Put the files to upload there, named <dataset name>.h5")
        print("(the conversion scripts write them there).")
        sys.exit(1)

    h5_files = sorted(DATA_SIA_DIR.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {DATA_SIA_DIR}")
        sys.exit(1)

    manifest = load_manifest()
    entries = {entry["name"]: entry for entry in manifest["datasets"]}

    # Build upload plan: only files whose content matches their manifest entry,
    # under the content name, and only if that name is not on the server yet
    plan = []
    for filepath in h5_files:
        dataset_name = filepath.stem  # strip .h5
        entry = entries.get(dataset_name)
        if entry is None:
            print(f"WARNING: Skipping {filepath.name} — not in datasets.json")
            continue
        if file_hash(filepath) != f"sha256:{entry['sha256']}":
            print(f"WARNING: Skipping {filepath.name} — sha256 differs from datasets.json (update the manifest first)")
            continue
        if object_exists(manifest, entry["filename"]):
            print(f"Skipping {dataset_name} — {entry['filename'][:16]}...h5 is already on the server")
            continue
        plan.append((filepath, dataset_name, entry["filename"]))

    # Print plan
    print(f"\n{'=' * 72}")
    print(f"Upload plan: {len(plan)} files from {DATA_SIA_DIR}")
    print(f"{'=' * 72}")
    for filepath, dataset_name, filename in plan:
        print(f"  {dataset_name:50s} -> {filename[:16]}...h5")
    print(f"{'=' * 72}\n")

    if not args.execute:
        print("DRY RUN — pass --execute to actually upload.")
        return

    # Upload each file. Nothing on the server is deleted or overwritten.
    s3_prefix = "s3://gadopt/g-drift/"
    for i, (filepath, dataset_name, filename) in enumerate(plan, 1):
        s3_dest = s3_prefix + filename
        print(f"[{i}/{len(plan)}] Uploading {dataset_name} -> {filename[:16]}...h5")
        result = s3cmd("put", "--acl-public", "--mime-type=application/x-hdf5", str(filepath), s3_dest)
        if result.returncode != 0:
            print(f"  FAILED to upload {dataset_name}")
            sys.exit(1)

    print(f"\nDone! Uploaded {len(plan)} files (public-read). Put their ETags into datasets.json.")


if __name__ == "__main__":
    main()
