#!/usr/bin/env python
"""Generate expected.pkl for geodynamic adiabat demo regression tests.

Run this script to regenerate expected values when the underlying
science or code intentionally changes.

Usage:
    python generate_expected.py
"""

import pickle
from pathlib import Path

import numpy as np
import gdrift


def main():
    demo_dir = Path(__file__).parent

    # Compute gravity and adiabats (same as demo.py)
    gravity_profile = gdrift.prem_gravity_profile()
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    slb24 = gdrift.ThermodynamicModel("SLB_24", "pyroliteCFMS")

    T0 = 1600
    adiabat_21 = gdrift.compute_adiabat(slb21, T0=T0, gravity_profile=gravity_profile)
    adiabat_24 = gdrift.compute_adiabat(slb24, T0=T0, gravity_profile=gravity_profile)

    # Sample at surface, ~500 km, ~1500 km, and CMB
    n = len(adiabat_21["depths"])
    test_indices = np.array([0, n // 5, n // 2, n - 1])

    expected = {
        "test_indices": test_indices,
        "slb21_temperature": adiabat_21["temperature"][test_indices],
        "slb24_temperature": adiabat_24["temperature"][test_indices],
        "slb21_density": adiabat_21["rho"][test_indices],
        "slb24_density": adiabat_24["rho"][test_indices],
        "slb21_Di": adiabat_21["Di"],
        "slb24_Di": adiabat_24["Di"],
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print(f"\nTest indices (depths): {adiabat_21['depths'][test_indices] / 1e3} km")
    print("\nExpected values:")
    for key, val in expected.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    main()
