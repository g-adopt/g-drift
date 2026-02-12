#!/usr/bin/env python
"""Generate expected.pkl for solidus demo regression tests.

Run this script to regenerate expected values when the underlying
science or code intentionally changes.

Usage:
    python generate_expected.py
"""

import pickle
from pathlib import Path

import numpy as np
import gdrift
from gdrift.profile import SplineProfile


def main():
    demo_dir = Path(__file__).parent

    # Load the solidus models
    fiquet_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Fiquet_et_al_2010_SCIENCE",
        description="Fiquet et al 2010 Science")

    andrault_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    hirsch_solidus = gdrift.HirschmannSolidus()

    # Create composite solidus (same logic as demo.py)
    my_depths = []
    my_solidus = []
    for solidus_model in [
        hirsch_solidus.get_profile("solidus temperature"),
        andrault_solidus.get_profile("solidus temperature"),
    ]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    ghelichkhan_et_al = SplineProfile(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021")

    # Define test depths (within valid ranges for each model)
    # Hirschmann: 0-303.9 km
    test_depths_upper = np.array([50e3, 150e3, 250e3])  # Upper mantle
    # Andrault: 389.3-2945.4 km, Fiquet: 831.5-2907.6 km
    test_depths_lower = np.array([1000e3, 1500e3, 2500e3])  # Lower mantle
    # Composite spans: 0 km to ~2940 km
    test_depths_full = np.array([100e3, 250e3, 500e3, 1500e3, 2500e3])  # Full range

    # Compute expected values
    expected = {
        "test_depths_upper": test_depths_upper,
        "test_depths_lower": test_depths_lower,
        "test_depths_full": test_depths_full,
        "hirsch_solidus": hirsch_solidus.get_profile("solidus temperature").at_depth(test_depths_upper),
        "andrault_solidus": andrault_solidus.get_profile("solidus temperature").at_depth(test_depths_lower),
        "fiquet_solidus": fiquet_solidus.get_profile("solidus temperature").at_depth(test_depths_lower),
        "composite_solidus": ghelichkhan_et_al.at_depth(test_depths_full),
    }

    # Save to pickle
    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print("\nExpected values:")
    for key, val in expected.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
