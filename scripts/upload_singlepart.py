#!/usr/bin/env python3
"""Upload datasets missing from S3 to Digital Ocean Spaces.

Reads datasets.json, checks which entries have no etag (i.e. not yet
on the server), and uploads them from the package directory (gdrift/data/)
under their content name (the `filename` field, `<sha256>.h5`). The local file
can have either the content name or the name-based name that the conversion
scripts write (`hash_name(name) + ".h5"`). Its sha256 must match the manifest.
An object that already exists on the server is never overwritten.

Usage:
    python scripts/upload_singlepart.py
"""
import json
import os
import sys

import boto3
import hashlib
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

with open('gdrift/datasets.json') as f:
    manifest = json.load(f)

# Read credentials from s3cmd config (no section headers)
creds = {}
with open(os.path.expanduser('~/.s3cfg-gadopt')) as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            creds[k.strip()] = v.strip()

s3 = boto3.client(
    's3',
    endpoint_url=f"https://{creds['host_base']}",
    aws_access_key_id=creds['access_key'],
    aws_secret_access_key=creds['secret_key'],
)

# Unsigned client for HEAD checks
s3_anon = boto3.client(
    's3',
    endpoint_url=f"https://{creds['host_base']}",
    config=Config(signature_version=UNSIGNED),
)

bucket = manifest['s3']['bucket']
prefix = manifest['s3']['prefix']

# Find datasets with no etag in the manifest
missing = [ds for ds in manifest['datasets'] if ds.get('etag') is None]

if not missing:
    print("All datasets have ETags — nothing to upload.")
    sys.exit(0)

print(f"Found {len(missing)} datasets without ETags to upload.\n")

for i, ds in enumerate(missing):
    name = ds['name']
    key = prefix + ds['filename']

    # Upload only under the content name
    if ds['filename'] != ds['sha256'] + '.h5':
        print(f'[{i+1}/{len(missing)}] {name}: SKIP (filename is not sha256 + ".h5")')
        continue

    # The local file can carry the content name or the name-based name
    hashed_name = hashlib.sha256(name.encode()).hexdigest() + '.h5'
    local_path = next((p for p in (f'gdrift/data/{ds["filename"]}', f'gdrift/data/{hashed_name}')
                       if os.path.exists(p)), None)
    if local_path is None:
        print(f'[{i+1}/{len(missing)}] {name}: SKIP (no local file)')
        continue

    # Upload only the content that the manifest describes
    h = hashlib.sha256()
    with open(local_path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != ds['sha256']:
        print(f'[{i+1}/{len(missing)}] {name}: SKIP (sha256 of {local_path} does not match the manifest)')
        continue

    # Never overwrite a file on the server: older releases may point to it
    # The server answers 404 for a missing key. Any other error (403,
    # throttling) is unexpected, so stop instead of risking an overwrite.
    try:
        s3_anon.head_object(Bucket=bucket, Key=key)
        print(f'[{i+1}/{len(missing)}] {name}: SKIP ({key} already exists)')
        continue
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') not in ('404', 'NoSuchKey', 'NotFound'):
            raise

    size_mb = os.path.getsize(local_path) / 1e6
    print(f'[{i+1}/{len(missing)}] {name} ({size_mb:.0f} MB)...', flush=True)

    try:
        s3.upload_file(local_path, bucket, key,
                       ExtraArgs={'ACL': 'public-read', 'ContentType': 'application/x-hdf5'})
    except Exception as e:
        print(f'  FAIL: {e}', flush=True)
        sys.exit(1)

    # Fetch the ETag and store it back in the manifest
    head = s3_anon.head_object(Bucket=bucket, Key=key)
    etag = head['ETag'].strip('"')
    ds['etag'] = etag
    print(f'  OK (etag={etag})', flush=True)

# Write updated manifest with new ETags
with open('gdrift/datasets.json', 'w') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')

print(f"\nDone! Uploaded {len(missing)} datasets and updated datasets.json.")
