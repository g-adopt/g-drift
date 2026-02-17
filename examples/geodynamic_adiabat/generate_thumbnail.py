#!/usr/bin/env python3
"""Generate thumbnail for the geodynamic adiabat example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gdrift

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    gravity = gdrift.prem_gravity_profile()
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    slb24 = gdrift.ThermodynamicModel("SLB_24", "pyroliteCFMS")

    adiabat_21 = gdrift.compute_adiabat(slb21, T0=1600, gravity_profile=gravity)
    adiabat_24 = gdrift.compute_adiabat(slb24, T0=1600, gravity_profile=gravity)

    depths_km = adiabat_21["depths"] / 1e3

    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5), sharey=True)
    specs = [
        ("temperature", "T [K]"),
        ("rho", r"$\rho$ [kg/m$^3$]"),
        ("alpha", r"$\alpha$ [1/K]"),
        ("Cp_SI", "Cp [J/kg/K]"),
        ("gravity", "g [m/s$^2$]"),
        ("Cv_SI", "Cv [J/kg/K]"),
    ]

    for ax, (key, xlabel) in zip(axes.flat, specs):
        ax.plot(adiabat_21[key], depths_km, linewidth=1, label="SLB_21")
        ax.plot(adiabat_24[key], depths_km, "--", linewidth=1, label="SLB_24")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.grid(alpha=0.2)
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)

    axes[0, 0].legend(fontsize=6)
    axes[0, 0].set_ylabel("Depth [km]", fontsize=7)
    axes[1, 0].set_ylabel("Depth [km]", fontsize=7)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "geodynamic_adiabat.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
