#!/usr/bin/env python3
"""Generate thumbnail for the linearisation (regularisation) example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gdrift

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    slb = gdrift.ThermodynamicModel(
        "SLB_21", "pyroliteCFMAS",
        temps=np.linspace(300, 4000),
        depths=np.linspace(0, 2890e3),
    )

    temp_profile = gdrift.SplineProfile(
        depth=np.asarray([0., 500e3, 2700e3, 3000e3]),
        value=np.asarray([300, 1000, 3000, 4000]),
    )

    regular = gdrift.regularise_thermodynamic_table(
        slb, temp_profile,
        regular_range={"v_s": (-1.5, 0.0), "v_p": (-np.inf, 0.0),
                       "rho": (-np.inf, 0.0)},
    )

    Vs_orig = slb.compute_swave_speed().get_vals()
    Vs_reg = regular.compute_swave_speed().get_vals()
    temperatures = slb.get_temperatures()
    all_depths = slb.get_depths()

    comparison_depths = np.array([410, 660, 1000, 2000]) * 1e3
    depth_indices = [np.abs(d - all_depths).argmin() for d in comparison_depths]

    fig, axes = plt.subplots(2, 2, figsize=(5, 4))

    for ax, idx in zip(axes.flat, depth_indices):
        d_km = all_depths[idx] / 1e3
        anchor_T = temp_profile.at_depth(all_depths[idx])
        ax.plot(temperatures, Vs_orig[idx, :], "b-", linewidth=0.8,
                label="Original")
        ax.plot(temperatures, Vs_reg[idx, :], "r-", linewidth=0.8,
                label="Regularised")
        ax.axvline(x=anchor_T, color="green", linestyle="--", linewidth=0.6)
        ax.set_title(f"{d_km:.0f} km", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=6)
    axes[1, 0].set_xlabel("Temperature [K]", fontsize=7)
    axes[1, 1].set_xlabel("Temperature [K]", fontsize=7)
    axes[0, 0].set_ylabel("Vs [m/s]", fontsize=7)
    axes[1, 0].set_ylabel("Vs [m/s]", fontsize=7)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "linearisation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
