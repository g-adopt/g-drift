#!/usr/bin/env python3
"""Generate thumbnail for the mantle solidus example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gdrift
from gdrift.profile import SplineProfile

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    fiquet = gdrift.RadialEarthModelFromFile(
        "1d_solidus_Fiquet_et_al_2010_SCIENCE")
    andrault = gdrift.RadialEarthModelFromFile(
        "1d_solidus_Andrault_et_al_2011_EPSL")
    hirsch = gdrift.HirschmannSolidus()

    # Composite solidus
    my_depths, my_solidus = [], []
    for profile in [hirsch.get_profile("solidus temperature"),
                    andrault.get_profile("solidus temperature")]:
        d_min, d_max = profile.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(profile.at_depth(dpths))
    composite = SplineProfile(
        depth=np.asarray(my_depths), value=np.asarray(my_solidus),
        name="Composite")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for model, ls in [(andrault.get_profile("solidus temperature"), "-"),
                      (hirsch.get_profile("solidus temperature"), "-."),
                      (fiquet.get_profile("solidus temperature"), "--"),
                      (composite, ":")]:
        d_min, d_max = model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        ax.plot(model.at_depth(dpths), dpths / 1e3, ls, linewidth=1.5,
                label=model.display_name)
    ax.invert_yaxis()
    ax.set_xlabel("Solidus Temperature [K]", fontsize=9)
    ax.set_ylabel("Depth [km]", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "mantle_solidus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
