#!/usr/bin/env python
"""Generate expected.pkl for temperature-to-velocity conversion demo.

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

    # Step 1: Load thermodynamic model
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")

    # Step 2: Load Terra temperature profile
    terra_data = np.loadtxt(
        demo_dir.parent / "TerraMT512vs.dat", unpack=False, usecols=(0, 1))
    temperature_profile = gdrift.SplineProfile(
        depth=terra_data[:, 0] * 1e3,
        value=terra_data[:, 1],
        name="Terra average temperature",
        extrapolate=True,
    )

    # Step 3: Regularise
    regular_slb21 = gdrift.regularise_thermodynamic_table(
        slb21,
        temperature_profile,
        regular_range={
            "v_s": (-1.5, 0.0),
            "v_p": (-np.inf, 0.0),
            "rho": (-np.inf, 0.0),
        },
    )

    # Step 4: Apply anelastic correction
    anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile("Q3")
    corrected_slb21 = gdrift.apply_anelastic_correction(
        regular_slb21, anelastic)

    # Step 5: Elastic and corrected Vs
    comparison_depths = np.array([200, 410, 660, 1000]) * 1e3
    test_temperatures = np.linspace(1000, 3500, 50)

    Vs_elastic = np.array([
        slb21.temperature_to_vs(test_temperatures, d)
        for d in comparison_depths
    ])
    Vs_corrected = np.array([
        corrected_slb21.temperature_to_vs(test_temperatures, d)
        for d in comparison_depths
    ])

    # Step 6: Load REVEAL and extract depth slice
    seismic_model = gdrift.SeismicModel("REVEAL")

    slice_depth = 200e3
    lats = np.linspace(-85, 85, 35)
    lons = np.linspace(-180, 175, 72)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    depth_grid = np.full_like(lat_grid, slice_depth)

    coordinates = gdrift.geodetic_to_cartesian(
        lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel())

    reveal_data = seismic_model.at(["vsh", "vsv"], coordinates)
    vsh = reveal_data[:, 0]
    vsv = reveal_data[:, 1]

    vs_isotropic = np.sqrt((2 * vsh**2 + vsv**2) / 3)

    # Step 7: Convert Vs -> T
    valid = np.isfinite(vs_isotropic)
    converted_temperature = np.full_like(vs_isotropic, np.nan)
    converted_temperature[valid] = corrected_slb21.vs_to_temperature(
        vs_isotropic[valid], depth_grid.ravel()[valid])

    # Build expected values
    expected = {
        "Vs_elastic": Vs_elastic,
        "Vs_corrected": Vs_corrected,
        "vs_iso_min": np.nanmin(vs_isotropic),
        "vs_iso_max": np.nanmax(vs_isotropic),
        "vs_iso_mean": np.nanmean(vs_isotropic),
        "temp_min": np.nanmin(converted_temperature),
        "temp_max": np.nanmax(converted_temperature),
        "temp_mean": np.nanmean(converted_temperature),
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    print(f"\nVs elastic shape: {Vs_elastic.shape}")
    print(f"Vs corrected shape: {Vs_corrected.shape}")
    print(f"\nREVEAL Vs (isotropic): "
          f"min={expected['vs_iso_min']:.1f}, "
          f"max={expected['vs_iso_max']:.1f}, "
          f"mean={expected['vs_iso_mean']:.1f} m/s")
    print(f"Converted T: "
          f"min={expected['temp_min']:.1f}, "
          f"max={expected['temp_max']:.1f}, "
          f"mean={expected['temp_mean']:.1f} K")


if __name__ == "__main__":
    main()
