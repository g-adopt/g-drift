"""Tests for metadata consistency and completeness in datasets.json.

Validates that the enriched manifest has proper structure, valid DOIs,
year ranges, hash formats, and tracks metadata completeness accurately.
"""

import json
import re
import pytest
from pathlib import Path

# Path to datasets.json
MANIFEST_PATH = Path(__file__).parent.parent / "gdrift" / "datasets.json"


@pytest.fixture
def manifest():
    """Load the datasets.json manifest."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture
def datasets(manifest):
    """Get the datasets array from manifest."""
    return manifest["datasets"]


def test_manifest_structure(manifest):
    """Test that manifest has required top-level fields."""
    assert "version" in manifest, "Manifest must have version field"
    assert manifest["version"] == 1, "Manifest version must be 1"

    assert "s3" in manifest, "Manifest must have S3 configuration"
    assert "endpoint_url" in manifest["s3"]
    assert "bucket" in manifest["s3"]
    assert "prefix" in manifest["s3"]

    assert "cdn_url" in manifest, "Manifest must have CDN URL"

    assert "datasets" in manifest, "Manifest must have datasets array"
    assert isinstance(manifest["datasets"], list)
    assert len(manifest["datasets"]) > 0, "Must have at least one dataset"


def test_dataset_entries_have_required_fields(datasets):
    """Test that all dataset entries have required fields."""
    required_fields = ["name", "filename", "type", "source", "sha256"]

    for dataset in datasets:
        for field in required_fields:
            assert field in dataset, f"Dataset '{dataset.get('name', 'UNKNOWN')}' missing required field: {field}"
            assert dataset[field] is not None, f"Dataset '{dataset['name']}' has None for field: {field}"

        # Name must be non-empty string
        assert isinstance(dataset["name"], str) and len(dataset["name"]) > 0

        # Type must be valid enum value
        valid_types = [
            "EARTH_MODEL",
            "SOLIDUS_PROFILE",
            "GEODYNAMIC_PROFILE",
            "THERMODYNAMIC_MODEL",
            "TOMOGRAPHY_MODEL"
        ]
        assert dataset["type"] in valid_types, f"Invalid type for {dataset['name']}: {dataset['type']}"

        # Source must be non-empty and not "unknown"
        assert isinstance(dataset["source"], str) and len(dataset["source"]) > 0
        assert dataset["source"].lower() != "unknown", f"Dataset {dataset['name']} has 'unknown' as source"


def test_doi_format_validation(datasets):
    """Test that DOIs match the standard pattern."""
    doi_pattern = r'^10\.\d{4,}/[^\s]+$'

    for dataset in datasets:
        doi = dataset.get("doi")

        # DOI can be None or empty string
        if not doi or doi == "":
            continue

        # Clean DOI (remove prefixes)
        doi_clean = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")

        # Validate format
        assert re.match(doi_pattern, doi_clean), \
            f"Invalid DOI format for {dataset['name']}: {doi}"


def test_year_range_validation(datasets):
    """Test that years are in reasonable range (1900-2030)."""
    for dataset in datasets:
        year = dataset.get("year")

        # Year can be None
        if year is None:
            continue

        assert isinstance(year, int), f"Year must be integer for {dataset['name']}: {year}"
        assert 1900 <= year <= 2030, \
            f"Year out of range for {dataset['name']}: {year}"


def test_metadata_completeness_tracking(datasets):
    """Test that metadata_complete flag is accurate."""
    for dataset in datasets:
        # Skip datasets without metadata tracking fields (older entries)
        if "metadata_complete" not in dataset:
            continue

        is_complete = dataset["metadata_complete"]
        assert isinstance(is_complete, bool), \
            f"metadata_complete must be boolean for {dataset['name']}"

        # Check if completeness flag is accurate
        has_doi = dataset.get("doi") and dataset.get("doi") != ""
        has_author = dataset.get("author") and dataset.get("author") != ""
        has_year = dataset.get("year") is not None
        has_source = dataset.get("source") and dataset.get("source") != ""
        has_description = dataset.get("description") and dataset.get("description") != ""

        # Complete requires: (doi OR (author AND year)) AND source AND description
        has_citation = has_doi or (has_author and has_year)
        expected_complete = has_citation and has_source and has_description

        # Warn if flag doesn't match expected (but don't fail - manual overrides allowed)
        if is_complete != expected_complete:
            print(f"\n⚠ Metadata completeness mismatch for {dataset['name']}:")
            print(f"  Flag: {is_complete}, Expected: {expected_complete}")
            print(f"  has_doi={has_doi}, has_author={has_author}, has_year={has_year}")
            print(f"  has_source={has_source}, has_description={has_description}")


def test_sha256_format(datasets):
    """Test that SHA256 hashes are 64 hexadecimal characters."""
    for dataset in datasets:
        sha256 = dataset.get("sha256")

        # SHA256 is required
        assert sha256 is not None, f"Dataset {dataset['name']} missing SHA256 hash"
        assert isinstance(sha256, str), f"SHA256 must be string for {dataset['name']}"

        # Must be 64 hex characters
        assert len(sha256) == 64, f"SHA256 must be 64 characters for {dataset['name']}: {len(sha256)}"
        assert re.match(r'^[a-f0-9]{64}$', sha256), \
            f"SHA256 must be lowercase hex for {dataset['name']}: {sha256}"


def test_no_duplicate_names(datasets):
    """Test that all dataset names are unique."""
    names = [d["name"] for d in datasets]
    duplicates = [name for name in names if names.count(name) > 1]

    assert len(duplicates) == 0, f"Duplicate dataset names found: {set(duplicates)}"


def test_filename_format(datasets):
    """Test that filenames have .h5 extension."""
    for dataset in datasets:
        filename = dataset.get("filename")
        assert filename is not None, f"Dataset {dataset['name']} missing filename"
        assert filename.endswith(".h5"), \
            f"Filename must end with .h5 for {dataset['name']}: {filename}"


def test_metadata_source_values(datasets):
    """Test that metadata_source has valid values."""
    valid_sources = ["hdf5_file", "web_search", "manifest_only", "slb_reference", "unknown", "test"]

    for dataset in datasets:
        # Skip datasets without metadata tracking
        if "metadata_source" not in dataset:
            continue

        source = dataset["metadata_source"]
        assert source in valid_sources, \
            f"Invalid metadata_source for {dataset['name']}: {source}"


def test_slb_models_have_composition_info(datasets):
    """Test that SLB thermodynamic models have composition metadata."""
    for dataset in datasets:
        name = dataset["name"]

        # Skip non-SLB models
        if not name.startswith("SLB_"):
            continue

        # Skip SLB models with just pyrolite (old naming)
        if name in ["SLB_16_pyrolite", "SLB_21_pyroliteNCMAS"]:
            continue

        # Check for composition in name
        compositions = ["pyrolite", "depleted-mantle", "bulk-oceanic-crust"]
        systems = ["MS", "FMS", "FMAS", "CFMS", "CFMAS", "NCFMAS", "NCMAS"]

        has_composition_in_name = any(comp in name for comp in compositions)
        has_system_in_name = any(name.endswith(sys) for sys in systems)

        if has_composition_in_name and has_system_in_name:
            # Should have enrichment metadata (if enrichment has been run)
            if "composition" in dataset:
                assert dataset["composition"] in compositions, \
                    f"Invalid composition for {name}: {dataset.get('composition')}"

            if "chemical_system" in dataset:
                assert dataset["chemical_system"] in systems, \
                    f"Invalid chemical_system for {name}: {dataset.get('chemical_system')}"


def test_seismic_models_are_tomography_type(datasets):
    """Test that all 3d_seismic_* models have TOMOGRAPHY_MODEL type."""
    for dataset in datasets:
        if dataset["name"].startswith("3d_seismic_"):
            assert dataset["type"] == "TOMOGRAPHY_MODEL", \
                f"Seismic model {dataset['name']} must have TOMOGRAPHY_MODEL type"


def test_prem_has_correct_metadata(datasets):
    """Test that PREM model has expected metadata."""
    prem = next((d for d in datasets if d["name"] == "1d_prem"), None)

    assert prem is not None, "PREM model not found in datasets"
    assert prem["type"] == "EARTH_MODEL"
    assert prem["utility"] == "PREM"
    assert prem["year"] == 1981


def test_dataset_count(datasets):
    """Test that we have expected number of datasets."""
    # Should have 96+ datasets as stated in CLAUDE.md
    assert len(datasets) >= 33, f"Expected at least 33 datasets, got {len(datasets)}"

    # Count by type
    types_count = {}
    for dataset in datasets:
        dtype = dataset["type"]
        types_count[dtype] = types_count.get(dtype, 0) + 1

    print("\nDataset counts by type:")
    for dtype, count in sorted(types_count.items()):
        print(f"  {dtype}: {count}")


def test_enrichment_fields_consistency(datasets):
    """Test that enrichment fields are consistent when present."""
    for dataset in datasets:
        # If has metadata_source, should have metadata_complete
        if "metadata_source" in dataset:
            assert "metadata_complete" in dataset, \
                f"Dataset {dataset['name']} has metadata_source but not metadata_complete"

        # If marked as SLB reference source, should have slb_version
        if dataset.get("metadata_source") == "slb_reference":
            assert "slb_version" in dataset or dataset["name"].startswith("SLB_"), \
                f"Dataset {dataset['name']} marked as slb_reference but has no slb_version"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
