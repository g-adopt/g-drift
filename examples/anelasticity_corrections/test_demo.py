"""Regression tests for the anelasticity demo."""

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


def test_elastic_vs(demo_namespace, expected_values):
    """Test that elastic Vs matches expected values."""
    np.testing.assert_allclose(
        demo_namespace["vs_elastic"],
        expected_values["vs_elastic"],
        rtol=1e-6,
    )


@pytest.mark.parametrize("model_key", [
    "Cammarano_Q1", "Cammarano_Q2", "Cammarano_Q3",
    "Cammarano_Q4", "Cammarano_Q5", "Cammarano_Q6",
    "Goes_Q4", "Goes_Q6",
])
def test_anelastic_vs(demo_namespace, expected_values, model_key):
    """Test that anelastic Vs for each Q-profile matches expected values."""
    np.testing.assert_allclose(
        demo_namespace["anelastic_vs"][model_key],
        expected_values["anelastic_vs"][model_key],
        rtol=1e-6,
    )


def test_velocity_reduction_positive(demo_namespace):
    """Anelastic corrections should always reduce Vs (positive reduction %)."""
    for model_key, reduction in demo_namespace["velocity_reduction"].items():
        assert np.all(reduction > 0), (
            f"{model_key}: expected all positive reductions, got {reduction}"
        )


def test_all_models_present(demo_namespace):
    """All 8 Q-profiles should be present in the combined dict."""
    expected_keys = {
        "Cammarano_Q1", "Cammarano_Q2", "Cammarano_Q3",
        "Cammarano_Q4", "Cammarano_Q5", "Cammarano_Q6",
        "Goes_Q4", "Goes_Q6",
    }
    assert set(demo_namespace["all_models"].keys()) == expected_keys
