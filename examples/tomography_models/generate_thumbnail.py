#!/usr/bin/env python3
"""Generate thumbnail for the tomography models example.

Uses S40RTS as a representative model on a Mollweide projection at 2700 km.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import gdrift

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    model = gdrift.SeismicModel("3d_seismic_S40RTS")
    depth = 2700e3

    lats = np.arange(-90, 91, 2)
    lons = np.arange(-180, 181, 2)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    depth_grid = np.full_like(lat_grid, depth, dtype=float)
    coords = gdrift.geodetic_to_cartesian(
        lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel())

    dvs = model.at("dvs", coords).reshape(lat_grid.shape)
    alpha = np.nanmax(np.abs(dvs))

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": "mollweide"})
    norm = TwoSlopeNorm(vmin=-alpha, vcenter=0, vmax=alpha)
    im = ax.pcolormesh(np.radians(lon_grid), np.radians(lat_grid), dvs,
                       cmap="RdBu", norm=norm, shading="auto")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"S40RTS at {depth/1e3:.0f} km", fontsize=9)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08,
                 label=r"$\delta V_s / V_s$")
    ax.tick_params(labelsize=6)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "tomography_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
