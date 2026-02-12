#!/usr/bin/env python
"""Generate expected.pkl for anelasticity demo regression tests.

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

    # Load elastic model
    slb_pyrolite = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")

    # Test coordinates
    test_depth = 500e3
    test_temps = np.array([1500.0, 2000.0, 2500.0, 3000.0])
    test_depths = np.full_like(test_temps, test_depth)

    # Elastic velocities
    vs_elastic = slb_pyrolite.temperature_to_vs(
        temperature=test_temps, depth=test_depths)

    # Cammarano Q-profiles
    cammarano_q_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    anelastic_vs = {}
    anelastic_vp = {}

    for qname in cammarano_q_names:
        anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile(qname)
        corrected = gdrift.apply_anelastic_correction(slb_pyrolite, anelastic)
        key = f"Cammarano_{qname}"
        anelastic_vs[key] = corrected.temperature_to_vs(
            temperature=test_temps, depth=test_depths)
        anelastic_vp[key] = corrected.temperature_to_vp(
            temperature=test_temps, depth=test_depths)

    # Goes Q-profiles
    goes_q_names = ["Q4", "Q6"]
    for qname in goes_q_names:
        anelastic = gdrift.GoesAnelasticityModel.from_q_profile(qname)
        corrected = gdrift.apply_anelastic_correction(slb_pyrolite, anelastic)
        key = f"Goes_{qname}"
        anelastic_vs[key] = corrected.temperature_to_vs(
            temperature=test_temps, depth=test_depths)
        anelastic_vp[key] = corrected.temperature_to_vp(
            temperature=test_temps, depth=test_depths)

    expected = {
        "test_depth": test_depth,
        "test_temps": test_temps,
        "vs_elastic": vs_elastic,
        "anelastic_vs": anelastic_vs,
        "anelastic_vp": anelastic_vp,
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print(f"\nElastic Vs: {vs_elastic}")
    for key in sorted(anelastic_vs):
        print(f"  {key} Vs: {anelastic_vs[key]}")


if __name__ == "__main__":
    main()
