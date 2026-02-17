"""Verify that every dataset on the S3 server matches its manifest ETag.

Run with:
    pytest tests/test_server_hashes.py -m server -v

Each test does a HEAD request to check the S3 ETag against the etag
field in datasets.json. No file downloads required — runs in seconds.
Requires network access, so excluded from normal CI via the ``server`` marker.
"""
import pytest

from gdrift.datasetnames import _load_manifest


def _get_s3_client(s3_config):
    """Create an unsigned boto3 S3 client."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=s3_config["endpoint_url"],
        config=Config(signature_version=UNSIGNED),
    )


def _build_dataset_params():
    """Build pytest parameters: one per dataset with an etag in the manifest."""
    manifest = _load_manifest()
    params = []
    for ds in manifest["datasets"]:
        etag = ds.get("etag")
        if etag is None:
            continue
        params.append(pytest.param(
            ds["name"], ds["filename"], etag,
            id=ds["name"],
        ))
    return params


@pytest.mark.server
@pytest.mark.parametrize("name,filename,expected_etag", _build_dataset_params())
def test_server_etag_matches_manifest(name, filename, expected_etag):
    """S3 ETag matches the etag field in datasets.json."""
    manifest = _load_manifest()
    s3_cfg = manifest["s3"]
    client = _get_s3_client(s3_cfg)

    key = s3_cfg["prefix"] + filename
    try:
        head = client.head_object(Bucket=s3_cfg["bucket"], Key=key)
    except Exception as e:
        pytest.fail(f"{name}: HEAD request failed: {e}")

    etag = head["ETag"].strip('"')
    assert etag == expected_etag, (
        f"{name}: server ETag {etag} != manifest {expected_etag}"
    )
