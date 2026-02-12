"""Regression tests for the solidus temperature demo."""

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


def test_hirschmann_solidus(demo_namespace, expected_values):
    """Test that Hirschmann solidus matches expected values."""
    hirsch_solidus = demo_namespace["hirsch_solidus"]
    profile = hirsch_solidus.get_profile("solidus temperature")

    test_depths = expected_values["test_depths_upper"]
    actual = profile.at_depth(test_depths)
    expected = expected_values["hirsch_solidus"]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_andrault_solidus(demo_namespace, expected_values):
    """Test that Andrault solidus matches expected values."""
    andrault_solidus = demo_namespace["andrault_solidus"]
    profile = andrault_solidus.get_profile("solidus temperature")

    test_depths = expected_values["test_depths_lower"]
    actual = profile.at_depth(test_depths)
    expected = expected_values["andrault_solidus"]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_fiquet_solidus(demo_namespace, expected_values):
    """Test that Fiquet solidus matches expected values."""
    fiquet_solidus = demo_namespace["fiquet_solidus"]
    profile = fiquet_solidus.get_profile("solidus temperature")

    test_depths = expected_values["test_depths_lower"]
    actual = profile.at_depth(test_depths)
    expected = expected_values["fiquet_solidus"]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_composite_solidus(demo_namespace, expected_values):
    """Test that the composite solidus matches expected values."""
    ghelichkhan_et_al = demo_namespace["ghelichkhan_et_al"]

    test_depths = expected_values["test_depths_full"]
    actual = ghelichkhan_et_al.at_depth(test_depths)
    expected = expected_values["composite_solidus"]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_depth_ranges(demo_namespace):
    """Test that depth ranges are as expected."""
    hirsch = demo_namespace["hirsch_solidus"].get_profile("solidus temperature")
    hirsch_min, hirsch_max = hirsch.min_max_depth()

    # Hirschmann covers upper mantle (0-304 km)
    assert hirsch_min < 10e3  # Starts at surface
    assert hirsch_max > 300e3  # Extends past 300 km

    composite = demo_namespace["ghelichkhan_et_al"]
    comp_min, comp_max = composite.min_max_depth()

    # Composite should span full mantle (0 to ~2940 km)
    assert comp_min < 10e3
    assert comp_max > 2900e3
