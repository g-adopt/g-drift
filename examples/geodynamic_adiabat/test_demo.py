"""Regression tests for the geodynamic adiabat demo."""

import pickle
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def demo_namespace():
    """Run the demo script and capture its namespace."""
    demo_dir = Path(__file__).parent
    namespace = {}
    exec(open(demo_dir / "demo.py").read(), namespace)
    return namespace


@pytest.fixture(scope="module")
def expected_values():
    """Load expected values from pickle file."""
    demo_dir = Path(__file__).parent
    with open(demo_dir / "expected.pkl", "rb") as f:
        return pickle.load(f)


def test_slb21_temperature(demo_namespace, expected_values):
    """Test SLB_21 adiabatic temperature at key depths."""
    actual = demo_namespace["adiabat_21"]["temperature"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["slb21_temperature"], rtol=1e-6)


def test_slb24_temperature(demo_namespace, expected_values):
    """Test SLB_24 adiabatic temperature at key depths."""
    actual = demo_namespace["adiabat_24"]["temperature"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["slb24_temperature"], rtol=1e-6)


def test_slb21_density(demo_namespace, expected_values):
    """Test SLB_21 density along the adiabat."""
    actual = demo_namespace["adiabat_21"]["rho"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["slb21_density"], rtol=1e-6)


def test_slb24_density(demo_namespace, expected_values):
    """Test SLB_24 density along the adiabat."""
    actual = demo_namespace["adiabat_24"]["rho"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["slb24_density"], rtol=1e-6)


def test_slb21_dissipation_number(demo_namespace, expected_values):
    """Test SLB_21 dissipation number."""
    actual = demo_namespace["adiabat_21"]["Di"]
    np.testing.assert_allclose(
        actual, expected_values["slb21_Di"], rtol=1e-6)


def test_slb24_dissipation_number(demo_namespace, expected_values):
    """Test SLB_24 dissipation number."""
    actual = demo_namespace["adiabat_24"]["Di"]
    np.testing.assert_allclose(
        actual, expected_values["slb24_Di"], rtol=1e-6)


def test_surface_gravity(demo_namespace):
    """Test gravity profile surface value."""
    g_surface = demo_namespace["gravity_profile"].at_depth(0)
    np.testing.assert_allclose(g_surface, 9.8, atol=0.2)


def test_temperature_monotonically_increasing(demo_namespace):
    """Test that temperature increases with depth for both models."""
    for key in ["adiabat_21", "adiabat_24"]:
        T = demo_namespace[key]["temperature"]
        assert np.all(np.diff(T) >= 0), f"{key} temperature not monotonically increasing"


def test_dissipation_number_range(demo_namespace):
    """Test that dissipation number is in a physically reasonable range."""
    # NOTE: Surface-based Di (~1.6) is higher than the commonly cited
    # depth-averaged value (~0.5-0.7). See CLAUDE.md for investigation note.
    for key in ["adiabat_21", "adiabat_24"]:
        Di = demo_namespace[key]["Di"]
        assert 0.3 < Di < 3.0, f"{key} Di={Di} outside expected range [0.3, 3.0]"
