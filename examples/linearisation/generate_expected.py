#!/usr/bin/env python
"""Generate expected.pkl for linearisation demo regression tests.

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

    # Reproduce the demo computation
    slb_pyrolite = gdrift.ThermodynamicModel(
        "SLB_21", "pyroliteCFMAS",
        temps=np.linspace(300, 4000),
        depths=np.linspace(0, 2890e3),
    )

    temperature_profile = gdrift.SplineProfile(
        depth=np.asarray([0., 500e3, 2700e3, 3000e3]),
        value=np.asarray([300, 1000, 3000, 4000]),
    )

    regular_slb = gdrift.regularise_thermodynamic_table(
        slb_pyrolite,
        temperature_profile,
        regular_range={
            "v_s": (-1.5, 0.0),
            "v_p": (-np.inf, 0.0),
            "rho": (-np.inf, 0.0),
        },
    )

    # Extract tables
    temperatures = slb_pyrolite.get_temperatures()
    all_depths = slb_pyrolite.get_depths()

    Vs_original = slb_pyrolite.compute_swave_speed().get_vals()
    Vp_original = slb_pyrolite.compute_pwave_speed().get_vals()
    rho_original = slb_pyrolite._tables["rho"].get_vals()

    Vs_regularised = regular_slb.compute_swave_speed().get_vals()
    Vp_regularised = regular_slb.compute_pwave_speed().get_vals()
    rho_regularised = regular_slb._tables["rho"].get_vals()

    # Test at four key depths
    comparison_depths = np.array([410, 660, 1000, 2000]) * 1e3
    depth_indices = np.array([np.abs(d - all_depths).argmin() for d in comparison_depths])

    # Sample at anchor temperatures for each depth
    anchor_t_indices = np.array([
        np.abs(temperatures - temperature_profile.at_depth(all_depths[idx])).argmin()
        for idx in depth_indices
    ])

    expected = {
        "depth_indices": depth_indices,
        "anchor_t_indices": anchor_t_indices,
        "Vs_original": Vs_original[depth_indices][:, anchor_t_indices],
        "Vp_original": Vp_original[depth_indices][:, anchor_t_indices],
        "rho_original": rho_original[depth_indices][:, anchor_t_indices],
        "Vs_regularised": Vs_regularised[depth_indices][:, anchor_t_indices],
        "Vp_regularised": Vp_regularised[depth_indices][:, anchor_t_indices],
        "rho_regularised": rho_regularised[depth_indices][:, anchor_t_indices],
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print(f"\nDepth indices: {depth_indices}")
    print(f"Depths (km): {all_depths[depth_indices] / 1e3}")
    print(f"Anchor T indices: {anchor_t_indices}")
    print(f"Anchor T (K): {temperatures[anchor_t_indices]}")
    print("\nExpected values (at anchor temperatures):")
    for key, val in expected.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}:\n    {val}")


if __name__ == "__main__":
    main()
