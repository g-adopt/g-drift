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


def _build_legacy_params():
    """Build pytest parameters for the name-based objects of gdrift 0.1.2 and 0.1.3.

    These releases load files named `hash_name(<dataset name>) + ".h5"`. The
    names and ETags are stored in tests/legacy_objects_v0.1.3.json, taken from
    the manifest at tag v0.1.3, because the current manifest only lists the
    content-named files.
    """
    import json
    from pathlib import Path

    path = Path(__file__).parent / "legacy_objects_v0.1.3.json"
    objects = json.loads(path.read_text())["objects"]
    return [
        pytest.param(entry["dataset"], filename, entry["etag"], id=entry["dataset"])
        for filename, entry in objects.items()
    ]


@pytest.mark.server
@pytest.mark.parametrize("name,filename,expected_etag", _build_legacy_params())
def test_server_keeps_objects_of_old_releases(name, filename, expected_etag):
    """The objects that gdrift 0.1.2 and 0.1.3 load are still on the server, unchanged.

    A dataset fix adds a new content-named object and must not touch these.
    A changed or missing ETag here means that an old release can no longer
    load this dataset.
    """
    manifest = _load_manifest()
    s3_cfg = manifest["s3"]
    client = _get_s3_client(s3_cfg)

    key = s3_cfg["prefix"] + filename
    try:
        head = client.head_object(Bucket=s3_cfg["bucket"], Key=key)
    except Exception as e:
        pytest.fail(f"{name}: HEAD request for the 0.1.3 object failed: {e}")

    etag = head["ETag"].strip('"')
    assert etag == expected_etag, (
        f"{name}: the 0.1.3 object changed on the server (ETag {etag} != {expected_etag})"
    )
