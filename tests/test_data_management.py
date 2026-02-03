import re
import pytest
import tempfile
import h5py
import numpy

from gdrift.datasetnames import (
    DATASET_REGISTRY,
    AVAILABLE_DATASETS,
    DatasetType,
    get_manifest_config,
    get_dataset_hash,
    _load_manifest,
)
from gdrift.seismic import AVAILABLE_SEISMIC_MODELS
from gdrift.io import load_dataset, file_hash, _verify_hash


def test_manifest_loads():
    """datasets.json is valid and all entries parse into Dataset objects."""
    manifest = _load_manifest()
    assert "version" in manifest
    assert "s3" in manifest
    assert "datasets" in manifest
    assert len(manifest["datasets"]) > 0
    # Every manifest entry has a corresponding Dataset in the registry
    for entry in manifest["datasets"]:
        assert entry["name"] in DATASET_REGISTRY


def test_manifest_hashes_valid():
    """All non-null hashes are 64-char hex strings (sha256)."""
    hex_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    for ds in AVAILABLE_DATASETS:
        if ds.file_hash is not None:
            assert hex_pattern.match(ds.file_hash), f"Invalid hash format for {ds.name}: {ds.file_hash}"


def test_registry_populated():
    """DATASET_REGISTRY contains all manifest entries."""
    manifest = _load_manifest()
    assert len(DATASET_REGISTRY) == len(manifest["datasets"])
    for entry in manifest["datasets"]:
        assert DATASET_REGISTRY.get_dataset(entry["name"]) is not None


def test_seismic_models_derived():
    """AVAILABLE_SEISMIC_MODELS matches tomography entries in the manifest."""
    tomography_datasets = DATASET_REGISTRY.filter_by_type(DatasetType.TOMOGRAPHY_MODEL)
    expected = [ds.name.replace("3d_seismic_", "") for ds in tomography_datasets]
    assert set(AVAILABLE_SEISMIC_MODELS) == set(expected)
    assert len(AVAILABLE_SEISMIC_MODELS) == len(tomography_datasets)


def test_load_rejects_unknown():
    """load_dataset raises ValueError for unknown datasets."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nonexistent_dataset_xyz")


def test_hash_verification():
    """Create a temp HDF5, verify hash, corrupt it, detect mismatch."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    # Create a valid HDF5 file
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("data", data=numpy.array([1.0, 2.0, 3.0]))

    # Compute its hash
    original_hash = file_hash(tmp_path)
    assert _verify_hash(tmp_path, original_hash)

    # Corrupt the file
    with open(tmp_path, "ab") as f:
        f.write(b"corruption")

    # Hash should now mismatch
    assert not _verify_hash(tmp_path, original_hash)

    # None hash always passes
    assert _verify_hash(tmp_path, None)

    import os
    os.unlink(tmp_path)


def test_s3_config():
    """get_manifest_config() returns expected keys."""
    config = get_manifest_config()
    assert "endpoint_url" in config
    assert "bucket" in config
    assert "prefix" in config
    assert "cdn_url" in config
    assert config["bucket"] == "gadopt"
    assert config["prefix"] == "g-drift/"


def test_get_dataset_hash_known():
    """get_dataset_hash returns a hash for known datasets."""
    h = get_dataset_hash("1d_prem")
    assert h is not None
    assert h.startswith("sha256:")


def test_get_dataset_hash_unknown():
    """get_dataset_hash returns None for unknown datasets."""
    h = get_dataset_hash("totally_bogus_dataset")
    assert h is None
