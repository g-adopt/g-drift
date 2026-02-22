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
from scipy.signal import savgol_filter


def main():
    demo_dir = Path(__file__).parent

    # Compute gravity and adiabat (same as demo.py)
    gravity_profile = gdrift.prem_gravity_profile()
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteFMS")

    T0 = 1600
    adiabat = gdrift.compute_adiabat(slb21, T0=T0, gravity_profile=gravity_profile)

    # Smoothed profiles
    smooth_keys = ["rho", "alpha", "Cp", "V", "Cv", "beta", "gamma"]
    adiabat_smooth = dict(adiabat)
    for key in smooth_keys:
        adiabat_smooth[key] = savgol_filter(adiabat[key], window_length=21, polyorder=3)
    adiabat_smooth["Cp_SI"] = adiabat_smooth["Cp"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])
    adiabat_smooth["Cv_SI"] = adiabat_smooth["Cv"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])

    # Sample at surface, ~500 km, ~1500 km, and CMB
    n = len(adiabat["depths"])
    test_indices = np.array([0, n // 5, n // 2, n - 1])

    expected = {
        "test_indices": test_indices,
        "temperature": adiabat["temperature"][test_indices],
        "density_raw": adiabat["rho"][test_indices],
        "density_smooth": adiabat_smooth["rho"][test_indices],
        "alpha_smooth": adiabat_smooth["alpha"][test_indices],
        "Cp_SI_smooth": adiabat_smooth["Cp_SI"][test_indices],
        "Di": adiabat["Di"],
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print(f"\nTest indices (depths): {adiabat['depths'][test_indices] / 1e3} km")
    print("\nExpected values:")
    for key, val in expected.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    main()
