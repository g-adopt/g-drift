"""Regression tests for the gadopt loading field demo.

These tests require gadopt (Firedrake) and are automatically skipped
when gadopt is not available.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

try:
    from gadopt import *  # noqa: F401, F403
    HAS_GADOPT = True
except ImportError:
    HAS_GADOPT = False

pytestmark = pytest.mark.skipif(
    not HAS_GADOPT, reason="gadopt/Firedrake not available")


@pytest.fixture(scope="module")
def demo_namespace():
    """Run the demo script and capture its namespace."""
    demo_dir = Path(__file__).parent
    namespace = {"__file__": str(demo_dir / "demo.py")}
    exec(open(demo_dir / "demo.py").read(), namespace)
    return namespace


@pytest.fixture(scope="module")
def expected_values():
    """Load expected values from pickle file."""
    demo_dir = Path(__file__).parent
    with open(demo_dir / "expected.pkl", "rb") as f:
        return pickle.load(f)


def test_mesh_vertices(demo_namespace, expected_values):
    """Test mesh has expected number of vertices."""
    n_vertices = len(demo_namespace["vs"].dat.data_with_halos)
    assert n_vertices == expected_values["n_vertices"]


def test_vs_range(demo_namespace, expected_values):
    """Test Vs values match expected range."""
    vs_data = demo_namespace["vs"].dat.data_with_halos
    np.testing.assert_allclose(
        vs_data.min(), expected_values["vs_min"], rtol=1e-4)
    np.testing.assert_allclose(
        vs_data.max(), expected_values["vs_max"], rtol=1e-4)


def test_vs_mean(demo_namespace, expected_values):
    """Test Vs mean value."""
    vs_data = demo_namespace["vs"].dat.data_with_halos
    np.testing.assert_allclose(
        vs_data.mean(), expected_values["vs_mean"], rtol=1e-4)


def test_temperature_range(demo_namespace, expected_values):
    """Test temperature values match expected range."""
    t_data = demo_namespace["temperature"].dat.data_with_halos
    np.testing.assert_allclose(
        t_data.min(), expected_values["temperature_min"], rtol=1e-4)
    np.testing.assert_allclose(
        t_data.max(), expected_values["temperature_max"], rtol=1e-4)


def test_temperature_mean(demo_namespace, expected_values):
    """Test temperature mean value."""
    t_data = demo_namespace["temperature"].dat.data_with_halos
    np.testing.assert_allclose(
        t_data.mean(), expected_values["temperature_mean"], rtol=1e-4)


def test_vs_positive(demo_namespace):
    """Test that all velocity values are positive."""
    vs_data = demo_namespace["vs"].dat.data_with_halos
    assert np.all(vs_data > 0), "All Vs values should be positive"


def test_temperature_physical_bounds(demo_namespace):
    """Test that temperatures are physically reasonable."""
    t_data = demo_namespace["temperature"].dat.data_with_halos
    assert t_data.min() > 200, "Temperature too low"
    assert t_data.max() < 6000, "Temperature too high"


def test_layer_averages_valid(demo_namespace):
    """Test that layer averages contain no NaN values."""
    v_ave = demo_namespace["v_ave"].dat.data_with_halos
    t_ave = demo_namespace["t_ave"].dat.data_with_halos
    assert not np.any(np.isnan(v_ave)), "Layer-averaged velocity has NaN"
    assert not np.any(np.isnan(t_ave)), "Layer-averaged temperature has NaN"
