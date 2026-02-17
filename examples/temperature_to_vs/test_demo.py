"""Regression tests for the temperature-to-velocity conversion demo."""

import pickle
from pathlib import Path

import numpy as np
import pytest


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


def test_vs_elastic(demo_namespace, expected_values):
    """Test elastic Vs at comparison depths and temperatures."""
    np.testing.assert_allclose(
        demo_namespace["Vs_elastic"],
        expected_values["Vs_elastic"], rtol=1e-6)


def test_vs_corrected(demo_namespace, expected_values):
    """Test anelastically-corrected Vs at comparison depths and temperatures."""
    np.testing.assert_allclose(
        demo_namespace["Vs_corrected"],
        expected_values["Vs_corrected"], rtol=1e-6)


def test_anelastic_reduces_velocity(demo_namespace):
    """Test that anelastic correction reduces Vs relative to regularised model."""
    corrected = demo_namespace["corrected_slb21"]
    regular = demo_namespace["regular_slb21"]
    comparison_depths = demo_namespace["comparison_depths"]
    test_temperatures = demo_namespace["test_temperatures"]

    for d in comparison_depths:
        Vs_reg = regular.temperature_to_vs(test_temperatures, d)
        Vs_cor = corrected.temperature_to_vs(test_temperatures, d)
        assert np.all(Vs_cor <= Vs_reg), \
            f"Corrected Vs should be <= regularised Vs at depth {d / 1e3:.0f} km"


def test_vs_isotropic_range(demo_namespace, expected_values):
    """Test REVEAL isotropic Vs statistics."""
    vs = demo_namespace["vs_isotropic"]
    np.testing.assert_allclose(
        np.nanmin(vs), expected_values["vs_iso_min"], rtol=1e-4)
    np.testing.assert_allclose(
        np.nanmax(vs), expected_values["vs_iso_max"], rtol=1e-4)
    np.testing.assert_allclose(
        np.nanmean(vs), expected_values["vs_iso_mean"], rtol=1e-4)


def test_converted_temperature_range(demo_namespace, expected_values):
    """Test converted temperature statistics."""
    T = demo_namespace["converted_temperature"]
    np.testing.assert_allclose(
        np.nanmin(T), expected_values["temp_min"], rtol=1e-3)
    np.testing.assert_allclose(
        np.nanmax(T), expected_values["temp_max"], rtol=1e-3)
    np.testing.assert_allclose(
        np.nanmean(T), expected_values["temp_mean"], rtol=1e-3)


def test_converted_temperature_physical(demo_namespace):
    """Test that converted temperatures are physically reasonable at 200 km."""
    T = demo_namespace["converted_temperature"]
    valid = np.isfinite(T)
    assert np.all(T[valid] > 500), "Temperatures should be > 500 K at 200 km"
    assert np.all(T[valid] < 3500), "Temperatures should be < 3500 K at 200 km"


def test_roundtrip_consistency(demo_namespace):
    """Test that T -> Vs -> T roundtrip is consistent at a test point."""
    corrected = demo_namespace["corrected_slb21"]
    test_T = 1800.0
    test_depth = 200e3

    vs = corrected.temperature_to_vs(test_T, test_depth)
    T_recovered = corrected.vs_to_temperature(vs, test_depth)

    np.testing.assert_allclose(
        T_recovered, test_T, rtol=1e-2,
        err_msg="T -> Vs -> T roundtrip should recover original temperature")
