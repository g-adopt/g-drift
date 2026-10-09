#!/usr/bin/env python3
"""Copy every dataset on the server to its content-based file name.

gdrift 0.1.3 and earlier name each file on the server after the SHA256 of the
dataset name (`hash_name(name) + ".h5"`). Current releases name each file
after the SHA256 of its content (the `filename` field of datasets.json), so
that a fixed dataset gets a new file and older releases keep their file.

This script makes a server-side copy of each name-based file to its content
name. It never deletes or overwrites a file: the name-based files stay, so
gdrift 0.1.3 and earlier keep working.

Modes (combine as needed; without a mode the script only checks and prints):

    python scripts/migrate_to_content_names.py
        Dry run. For each dataset, check with unsigned HEAD requests that the
        name-based file exists and that its ETag equals the `etag` in
        datasets.json (so its content is the content that the sha256 in
        datasets.json describes). Print the s3cmd copy command for each content
        name that does not exist yet.

    python scripts/migrate_to_content_names.py --execute
        Run those copy commands with s3cmd and ~/.s3cfg-gadopt.

    python scripts/migrate_to_content_names.py --update-etags
        Read the ETag of each content-named file and write it into the `etag`
        field of datasets.json. A server-side copy can change the ETag (for
        example for files that were uploaded in several parts), and
        tests/test_server_hashes.py compares these ETags.

    python scripts/migrate_to_content_names.py --verify-download
        Download each content-named file through the CDN and check its SHA256
        against datasets.json. This downloads every dataset (several GB).

A dataset whose name-based file does not have the expected ETag is not copied.
Rebuild and upload such a dataset under its content name by hand, as described
in docs/dataset-releases.md.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the hash_name function from the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gdrift.datasetnames import hash_name  # noqa: E402

S3CMD_CONFIG = Path.home() / ".s3cfg-gadopt"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "gdrift" / "datasets.json"


def origin_url(manifest, key):
    """Return the path-style HTTPS URL of an object on the server itself (not the CDN).

    The ETag is read from the server itself, because the CDN can add or
    change headers.
    """
    s3 = manifest["s3"]
    return f"{s3['endpoint_url'].rstrip('/')}/{s3['bucket']}/{key}"


def head_etag(url):
    """Return the ETag of a public object (without quotes), or None if it does not exist."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.headers["ETag"].strip('"')
    except urllib.error.HTTPError as e:
        # The server answers 404 for a missing key. Any other error (403,
        # throttling) is unexpected, so stop instead of guessing.
        if e.code == 404:
            return None
        raise


def sha256_of_url(url):
    """Download a URL in chunks and return the hex SHA256 of its content."""
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="run the s3cmd copy commands")
    parser.add_argument("--update-etags", action="store_true", help="write the ETags of the content-named files into datasets.json")
    parser.add_argument("--verify-download", action="store_true", help="download every content-named file through the CDN and check its SHA256")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    prefix = manifest["s3"]["prefix"]
    bucket = manifest["s3"]["bucket"]

    # Group datasets by content name: two datasets with identical content share one file
    by_filename = {}
    for entry in manifest["datasets"]:
        expected = entry["sha256"] + ".h5"
        if entry["filename"] != expected:
            sys.exit(f"{entry['name']}: filename {entry['filename']} is not sha256 + '.h5'")
        by_filename.setdefault(entry["filename"], []).append(entry)

    to_copy = []      # (source key, destination key, dataset names)
    problems = []     # datasets that cannot be copied safely
    present = 0
    for filename, entries in by_filename.items():
        names = ", ".join(e["name"] for e in entries)
        dst_key = prefix + filename
        if head_etag(origin_url(manifest, dst_key)) is not None:
            present += 1
            continue
        # Find a name-based source whose ETag matches the manifest, so its
        # content is the content that the sha256 (and so the new name) describes
        source = None
        for e in entries:
            src_key = prefix + hash_name(e["name"]) + ".h5"
            etag = head_etag(origin_url(manifest, src_key))
            if etag is not None and etag == e.get("etag"):
                source = src_key
                break
        if source is None:
            problems.append(f"{names}: no name-based file with the manifest ETag")
            continue
        to_copy.append((source, dst_key, names))

    print(f"{len(by_filename)} content names: {present} already on the server, "
          f"{len(to_copy)} to copy, {len(problems)} with problems.")
    for p in problems:
        print(f"  PROBLEM {p}")

    # The copy commands, in the form used in docs/dataset-releases.md
    commands = [
        ["s3cmd", "-c", str(S3CMD_CONFIG), "cp", "--acl-public",
         f"s3://{bucket}/{src}", f"s3://{bucket}/{dst}"]
        for src, dst, _ in to_copy
    ]
    for (src, dst, names), cmd in zip(to_copy, commands):
        print(f"\n# {names}\n{' '.join(cmd)}")

    if args.execute:
        for (src, dst, names), cmd in zip(to_copy, commands):
            print(f"Copying {names} ...", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                sys.exit(f"FAILED for {names}: {result.stderr.strip()}")
        print(f"Copied {len(to_copy)} files.")
    elif to_copy:
        print("\nDRY RUN: pass --execute to run these commands.")

    if args.update_etags:
        # Read every ETag first, and write the manifest only if all
        # content-named files exist, so a partial migration is never recorded
        etags = {}
        missing = []
        for entry in manifest["datasets"]:
            etag = head_etag(origin_url(manifest, prefix + entry["filename"]))
            if etag is None:
                missing.append(entry["name"])
            else:
                etags[entry["name"]] = etag
        if missing:
            sys.exit(f"Not updating {MANIFEST_PATH}: missing on the server: {', '.join(missing)}")
        for entry in manifest["datasets"]:
            entry["etag"] = etags[entry["name"]]
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Updated the ETags of {len(etags)} datasets in {MANIFEST_PATH}.")

    if args.verify_download:
        bad = []
        for filename, entries in by_filename.items():
            names = ", ".join(e["name"] for e in entries)
            # Report a download error for this file and go on with the next
            try:
                digest = sha256_of_url(manifest["cdn_url"] + filename)
                status = "OK" if digest == entries[0]["sha256"] else "MISMATCH"
            except (urllib.error.URLError, OSError) as e:
                status = f"ERROR ({e})"
            print(f"  {status} {names}", flush=True)
            if status != "OK":
                bad.append(names)
        print(f"Verified {len(by_filename)} files through the CDN, {len(bad)} mismatches or errors.")
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
