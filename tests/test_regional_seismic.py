"""Tests for regional seismic model support and GRD-collection integration.

Uses synthetic data (no network required) to verify:
- EarthModel3D returns NaN for queries outside regional coverage
- EarthModel3D returns valid values inside the coverage area
- New model names are present in AVAILABLE_SEISMIC_MODELS
- Overlapping model fields are updated correctly
- Regional flag is preserved in the manifest/registry
"""

import numpy as np
import pytest

from gdrift.earthmodel3d import EarthModel3D
from gdrift.constants import R_earth


# ── Helpers ─────────────────────────────────────────────────────────────────

def _geodetic_to_cartesian(lat_deg, lon_deg, depth_m):
    """Convert geographic (lat, lon, depth) to Cartesian (x, y, z)."""
    r = R_earth - depth_m
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def _build_regional_model(lat_min, lat_max, lon_min, lon_max,
                          depth_km=400, grid_spacing=5.0):
    """Build a synthetic EarthModel3D with data only inside a geographic box.

    Points outside the box have NaN values; points inside have value 4500.0.
    """
    model = EarthModel3D(nearest_neighbours=4, default_max_distance=500e3)

    # Create a global Fibonacci-like grid at a single depth
    lats = np.arange(-90, 91, grid_spacing)
    lons = np.arange(-180, 181, grid_spacing)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    depth_m = depth_km * 1e3
    coords = np.array([
        _geodetic_to_cartesian(lat, lon, depth_m)
        for lat, lon in zip(lat_flat, lon_flat)
    ])

    model.set_coordinates(coords[:, 0], coords[:, 1], coords[:, 2])

    # Values: 4500 inside the box, NaN outside
    inside = (
        (lat_flat >= lat_min) & (lat_flat <= lat_max) &
        (lon_flat >= lon_min) & (lon_flat <= lon_max)
    )
    values = np.full(len(lat_flat), np.nan)
    values[inside] = 4500.0

    model.add_quantity("vsv", values)
    return model


# ── Tests ───────────────────────────────────────────────────────────────────

class TestRegionalModelNaNBehavior:
    """Test that EarthModel3D correctly handles NaN in regional data."""

    @pytest.fixture
    def africa_model(self):
        """Regional model covering roughly Africa: lat -35..35, lon -20..55."""
        return _build_regional_model(-35, 35, -20, 55)

    def test_regional_model_returns_nan_outside_coverage(self, africa_model):
        """Points far outside the regional box should return NaN."""
        # Query in North America (lat=40, lon=-100) at 400 km depth
        query = _geodetic_to_cartesian(40, -100, 400e3).reshape(1, 3)
        result = africa_model.at("vsv", query)
        assert np.isnan(result).all(), (
            f"Expected NaN outside coverage, got {result}")

    def test_regional_model_returns_valid_inside(self, africa_model):
        """Points inside the regional box should return valid (non-NaN) values."""
        # Query in central Africa (lat=0, lon=25) at 400 km depth
        query = _geodetic_to_cartesian(0, 25, 400e3).reshape(1, 3)
        result = africa_model.at("vsv", query)
        assert not np.isnan(result).any(), (
            f"Expected valid values inside coverage, got {result}")
        assert np.abs(result - 4500.0) < 100, (
            f"Expected ~4500, got {result}")


class TestRegistryIntegration:
    """Tests for manifest/registry consistency with GRD models."""

    def test_new_models_in_registry(self):
        """All 22 new GRD model names should be present in the seismic models list."""
        from gdrift.seismic import AVAILABLE_SEISMIC_MODELS

        new_models = [
            "3D2015-07Sv", "AF2019", "ANT-20", "AuSREM", "Aus22",
            "CAM2016", "CSEM-2019", "DETOX-P02", "F2010-Afr", "FR12",
            "LLNL-G3D-JPS", "MITS-18", "PM13", "PMEAN", "SA2019",
            "SAVANI", "SEMUM2", "SL2013NA", "SL2013sv",
            "SL2013sv-uninterp", "SMEAN", "Y14",
        ]

        for name in new_models:
            assert name in AVAILABLE_SEISMIC_MODELS, (
                f"{name} not found in AVAILABLE_SEISMIC_MODELS. "
                f"Has the manifest been updated?")

    def test_overlapping_fields_updated(self):
        """Overlapping models should have their fields updated to absolute-only."""
        from gdrift.datasetnames import DATASET_REGISTRY

        # S40RTS should have vs/vsh/vsv, NOT dvs
        s40rts = DATASET_REGISTRY.get_dataset("3d_seismic_S40RTS")
        assert s40rts is not None, "S40RTS not in registry"
        assert "dvs" not in (s40rts.fields or []), (
            "S40RTS should no longer contain 'dvs'")
        assert set(s40rts.fields or []) == {"vs", "vsh", "vsv"}, (
            f"S40RTS fields should be {{vs, vsh, vsv}}, got {s40rts.fields}")

        # SPani should have absolute fields, not dvp/dvs/phi/xi
        spani = DATASET_REGISTRY.get_dataset("3d_seismic_SPani")
        assert spani is not None
        expected = {"vp", "vph", "vpv", "vs", "vsh", "vsv"}
        assert set(spani.fields or []) == expected, (
            f"SPani fields expected {expected}, got {spani.fields}")

    def test_regional_flag_in_manifest(self):
        """Regional models should have regional=True in the registry."""
        from gdrift.datasetnames import DATASET_REGISTRY

        regional_models = [
            "AF2019", "ANT-20", "AuSREM", "Aus22",
            "F2010-Afr", "FR12", "MITS-18", "SA2019", "Y14",
        ]

        for name in regional_models:
            dataset_name = f"3d_seismic_{name}"
            ds = DATASET_REGISTRY.get_dataset(dataset_name)
            assert ds is not None, f"{dataset_name} not in registry"
            assert ds.regional is True, (
                f"{dataset_name} should have regional=True, got {ds.regional}")

    def test_non_regional_models_no_flag(self):
        """Non-regional models should not have regional=True."""
        from gdrift.datasetnames import DATASET_REGISTRY

        non_regional = ["CAM2016", "SAVANI", "SMEAN", "DETOX-P02"]
        for name in non_regional:
            dataset_name = f"3d_seismic_{name}"
            ds = DATASET_REGISTRY.get_dataset(dataset_name)
            if ds is not None:
                assert ds.regional is not True, (
                    f"{dataset_name} should not be regional, got {ds.regional}")

    def test_total_seismic_model_count(self):
        """After integration, we should have 47 seismic models."""
        from gdrift.seismic import AVAILABLE_SEISMIC_MODELS
        assert len(AVAILABLE_SEISMIC_MODELS) == 47, (
            f"Expected 47 seismic models, got {len(AVAILABLE_SEISMIC_MODELS)}")
