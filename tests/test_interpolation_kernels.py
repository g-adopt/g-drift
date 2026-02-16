#!/usr/bin/env python3
"""
Tests for the interpolation kernels in EarthModel3D.

Constructs a synthetic 3D model with a known analytic field and verifies
that each kernel returns sensible results: correct shape, finite values,
exact recovery at data points, and smooth behaviour away from them.
"""

import numpy as np
import pytest
import gdrift


@pytest.fixture
def synthetic_model():
    """Build a synthetic EarthModel3D on a regular lat/lon/depth grid
    with a smooth analytic field: f = cos(lat_rad) * sin(lon_rad)."""
    lats = np.linspace(-80, 80, 17)
    lons = np.linspace(0, 350, 36)
    depths = np.array([100e3, 500e3, 1000e3])

    lat_grid, lon_grid, dep_grid = np.meshgrid(lats, lons, depths, indexing='ij')
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    dep_flat = dep_grid.ravel()

    coords = gdrift.geodetic_to_cartesian(lat_flat, lon_flat, dep_flat)

    # Smooth analytic field
    values = np.cos(np.radians(lat_flat)) * np.sin(np.radians(lon_flat))

    model = gdrift.EarthModel3D(nearest_neighbours=8, default_max_distance=2000e3)
    model.set_coordinates(*coords.T)
    model.add_quantity("test_field", values)
    return model


KERNELS = [
    {"kernel": "idw"},
    {"kernel": "gaussian"},
    {"kernel": "gaussian", "sigma": 50000},
    {"kernel": "idw_power", "power": 3.0},
    {"kernel": "exponential", "decay_length": 100000},
    {"kernel": "wendland", "support_radius": 200000},
]

KERNEL_IDS = [
    "idw",
    "gaussian_adaptive",
    "gaussian_fixed",
    "idw_power3",
    "exponential",
    "wendland",
]


@pytest.mark.parametrize("kernel_params", KERNELS, ids=KERNEL_IDS)
def test_kernel_returns_correct_shape(synthetic_model, kernel_params):
    """Each kernel should return one value per query point."""
    query_coords = gdrift.geodetic_to_cartesian(
        np.array([0.0, 30.0, -45.0]),
        np.array([90.0, 180.0, 270.0]),
        np.array([300e3, 300e3, 300e3]),
    )
    result = synthetic_model.at(label="test_field", coordinates=query_coords, **kernel_params)
    assert result.shape == (3,)


@pytest.mark.parametrize("kernel_params", KERNELS, ids=KERNEL_IDS)
def test_kernel_returns_finite_values(synthetic_model, kernel_params):
    """All interpolated values should be finite (no NaN or Inf)."""
    lats = np.linspace(-60, 60, 7)
    lons = np.linspace(10, 340, 7)
    lat_q, lon_q = np.meshgrid(lats, lons, indexing='ij')
    dep_q = np.full_like(lat_q, 500e3)
    query_coords = gdrift.geodetic_to_cartesian(lat_q.ravel(), lon_q.ravel(), dep_q.ravel())

    result = synthetic_model.at(label="test_field", coordinates=query_coords, **kernel_params)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("kernel_params", KERNELS, ids=KERNEL_IDS)
def test_kernel_values_within_data_range(synthetic_model, kernel_params):
    """Interpolated values should stay within [-1, 1] since the analytic
    field is cos(lat)*sin(lon) which is bounded by [-1, 1]."""
    lats = np.linspace(-60, 60, 5)
    lons = np.linspace(10, 340, 5)
    lat_q, lon_q = np.meshgrid(lats, lons, indexing='ij')
    dep_q = np.full_like(lat_q, 500e3)
    query_coords = gdrift.geodetic_to_cartesian(lat_q.ravel(), lon_q.ravel(), dep_q.ravel())

    result = synthetic_model.at(label="test_field", coordinates=query_coords, **kernel_params)
    assert np.all(result >= -1.1), f"Values below -1.1: {result.min()}"
    assert np.all(result <= 1.1), f"Values above 1.1: {result.max()}"


@pytest.mark.parametrize("kernel_params", KERNELS, ids=KERNEL_IDS)
def test_kernel_recovers_at_data_points(synthetic_model, kernel_params):
    """Querying exactly at a grid node should recover the analytic value."""
    lat, lon, depth = 0.0, 90.0, 500e3
    expected = np.cos(np.radians(lat)) * np.sin(np.radians(lon))  # = 1.0

    query_coords = gdrift.geodetic_to_cartesian(
        np.array([lat]), np.array([lon]), np.array([depth])
    )
    result = synthetic_model.at(label="test_field", coordinates=query_coords, **kernel_params)
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_unknown_kernel_raises(synthetic_model):
    """Requesting a non-existent kernel should raise ValueError."""
    query_coords = gdrift.geodetic_to_cartesian(
        np.array([0.0]), np.array([0.0]), np.array([500e3])
    )
    with pytest.raises(ValueError, match="Unknown kernel"):
        synthetic_model.at(label="test_field", coordinates=query_coords, kernel="cubic_magic")


def test_kernels_produce_different_results(synthetic_model):
    """Different kernels should generally give slightly different values
    at non-grid points, confirming they're not all identical."""
    query_coords = gdrift.geodetic_to_cartesian(
        np.array([15.0]), np.array([135.0]), np.array([300e3])
    )
    results = []
    for kp in KERNELS:
        val = synthetic_model.at(label="test_field", coordinates=query_coords, **kp)
        results.append(float(val))

    # Not all kernels should give the exact same number
    assert len(set(f"{v:.6f}" for v in results)) > 1, "All kernels returned identical values"
