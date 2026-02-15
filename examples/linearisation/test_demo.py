"""Regression tests for the linearisation demo."""

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


def test_vs_original(demo_namespace, expected_values):
    """Test original Vs values at key depths and anchor temperatures."""
    actual = demo_namespace["Vs_original"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["Vs_original"], rtol=1e-6)


def test_vp_original(demo_namespace, expected_values):
    """Test original Vp values at key depths and anchor temperatures."""
    actual = demo_namespace["Vp_original"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["Vp_original"], rtol=1e-6)


def test_rho_original(demo_namespace, expected_values):
    """Test original density values at key depths and anchor temperatures."""
    actual = demo_namespace["rho_original"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["rho_original"], rtol=1e-6)


def test_vs_regularised(demo_namespace, expected_values):
    """Test regularised Vs values at key depths and anchor temperatures."""
    actual = demo_namespace["Vs_regularised"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["Vs_regularised"], rtol=1e-6)


def test_vp_regularised(demo_namespace, expected_values):
    """Test regularised Vp values at key depths and anchor temperatures."""
    actual = demo_namespace["Vp_regularised"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["Vp_regularised"], rtol=1e-6)


def test_rho_regularised(demo_namespace, expected_values):
    """Test regularised density values at key depths and anchor temperatures."""
    actual = demo_namespace["rho_regularised"]
    di = expected_values["depth_indices"]
    ti = expected_values["anchor_t_indices"]
    np.testing.assert_allclose(
        actual[di][:, ti], expected_values["rho_regularised"], rtol=1e-6)


def test_vs_gradient_regularised(demo_namespace):
    """Test that regularised Vs has no large positive temperature gradients."""
    Vs = demo_namespace["Vs_regularised"]
    temperatures = demo_namespace["temperatures"]
    dVs_dT = np.gradient(Vs, temperatures, axis=1)
    # Allow a small tolerance for numerical noise
    assert np.all(dVs_dT < 0.1), \
        "Regularised Vs should have no significant positive dVs/dT"


def test_anchor_agreement(demo_namespace):
    """Test that original and regularised models agree near the anchor T."""
    Vs_orig = demo_namespace["Vs_original"]
    Vs_reg = demo_namespace["Vs_regularised"]
    temperatures = demo_namespace["temperatures"]
    all_depths = demo_namespace["all_depths"]
    temperature_profile = demo_namespace["temperature_profile"]

    comparison_depths = np.array([410, 660, 1000, 2000]) * 1e3
    for d in comparison_depths:
        d_idx = np.abs(d - all_depths).argmin()
        anchor_T = temperature_profile.at_depth(all_depths[d_idx])
        t_idx = np.abs(temperatures - anchor_T).argmin()
        np.testing.assert_allclose(
            Vs_reg[d_idx, t_idx], Vs_orig[d_idx, t_idx], rtol=1e-2,
            err_msg=f"Vs mismatch at anchor T for depth {d/1e3:.0f} km")
