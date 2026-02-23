#!/usr/bin/env python3
"""Generate thumbnail for the temperature-to-Vs conversion example.

Shows the REVEAL Vs map and converted temperature at 200 km depth.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gdrift

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    _demo_dir = Path(__file__).parent

    # Build the corrected model
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    terra_data = np.loadtxt(
        _demo_dir.parent / "TerraMT512vs.dat", unpack=False, usecols=(0, 1))
    temp_profile = gdrift.SplineProfile(
        depth=terra_data[:, 0] * 1e3, value=terra_data[:, 1],
        name="Terra", extrapolate=True)
    regular = gdrift.regularise_thermodynamic_table(
        slb21, temp_profile,
        regular_range={"v_s": (-1.5, 0.0), "v_p": (-np.inf, 0.0),
                       "rho": (-np.inf, 0.0)})
    anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile("Q3")
    corrected = gdrift.apply_anelastic_correction(regular, anelastic)

    # Query REVEAL at 200 km
    seismic = gdrift.SeismicModel("REVEAL")
    slice_depth = 200e3
    lats = np.linspace(-85, 85, 35)
    lons = np.linspace(-180, 175, 72)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    depth_grid = np.full_like(lat_grid, slice_depth)
    coords = gdrift.geodetic_to_cartesian(
        lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel())
    data = seismic.at(["vsh", "vsv"], coords)
    vs_iso = np.sqrt((2 * data[:, 0]**2 + data[:, 1]**2) / 3)

    # Convert to temperature
    valid = np.isfinite(vs_iso)
    temperature = np.full_like(vs_iso, np.nan)
    temperature[valid] = corrected.vs_to_temperature(
        vs_iso[valid], depth_grid.ravel()[valid])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    vs_map = vs_iso.reshape(lat_grid.shape)
    im1 = ax1.pcolormesh(lon_grid, lat_grid, vs_map,
                         cmap="seismic_r", shading="auto")
    fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    ax1.set_title("REVEAL Vs [m/s]", fontsize=8)
    ax1.tick_params(labelsize=6)

    temp_map = temperature.reshape(lat_grid.shape)
    im2 = ax2.pcolormesh(lon_grid, lat_grid, temp_map,
                         cmap="hot", shading="auto")
    fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    ax2.set_title("Temperature [K]", fontsize=8)
    ax2.tick_params(labelsize=6)

    fig.suptitle("Vs-to-temperature at 200 km", fontsize=9)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "temperature_to_vs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
