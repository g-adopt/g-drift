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


def test_temperature(demo_namespace, expected_values):
    """Test adiabatic temperature at key depths."""
    actual = demo_namespace["adiabat"]["temperature"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["temperature"], rtol=1e-6)


def test_density_raw(demo_namespace, expected_values):
    """Test raw density along the adiabat."""
    actual = demo_namespace["adiabat"]["rho"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["density_raw"], rtol=1e-6)


def test_density_smooth(demo_namespace, expected_values):
    """Test smoothed density along the adiabat."""
    actual = demo_namespace["adiabat_smooth"]["rho"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["density_smooth"], rtol=1e-6)


def test_alpha_smooth(demo_namespace, expected_values):
    """Test smoothed thermal expansivity at key depths."""
    actual = demo_namespace["adiabat_smooth"]["alpha"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["alpha_smooth"], rtol=1e-6)


def test_Cp_SI_smooth(demo_namespace, expected_values):
    """Test smoothed specific heat capacity at key depths."""
    actual = demo_namespace["adiabat_smooth"]["Cp_SI"]
    indices = expected_values["test_indices"]
    np.testing.assert_allclose(
        actual[indices], expected_values["Cp_SI_smooth"], rtol=1e-6)


def test_dissipation_number(demo_namespace, expected_values):
    """Test dissipation number."""
    actual = demo_namespace["adiabat"]["Di"]
    np.testing.assert_allclose(
        actual, expected_values["Di"], rtol=1e-6)


def test_surface_gravity(demo_namespace):
    """Test gravity profile surface value."""
    g_surface = demo_namespace["gravity_profile"].at_depth(0)
    np.testing.assert_allclose(g_surface, 9.8, atol=0.2)


def test_temperature_monotonically_increasing(demo_namespace):
    """Test that temperature increases with depth."""
    T = demo_namespace["adiabat"]["temperature"]
    assert np.all(np.diff(T) >= 0), "Temperature not monotonically increasing"


def test_dissipation_number_range(demo_namespace):
    """Test that dissipation number is in a physically reasonable range."""
    Di = demo_namespace["adiabat"]["Di"]
    assert 0.3 < Di < 3.0, f"Di={Di} outside expected range [0.3, 3.0]"
