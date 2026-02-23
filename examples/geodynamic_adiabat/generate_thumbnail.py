geodynamic_adiabat/generate_expected.py#!/usr/bin/env python3
"""Generate thumbnail for the geodynamic adiabat example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import gdrift
from gdrift.adiabat import prem_gravity_profile

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    gravity = prem_gravity_profile()
    slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteFMS")

    adiabat = gdrift.compute_adiabat(slb21, T0=1600, gravity_profile=gravity)

    # Smooth phase-transition spikes
    smooth_keys = ["rho", "alpha", "Cp", "V", "Cv", "beta", "gamma"]
    adiabat_smooth = dict(adiabat)
    for key in smooth_keys:
        adiabat_smooth[key] = savgol_filter(adiabat[key], window_length=21, polyorder=3)
    adiabat_smooth["Cp_SI"] = adiabat_smooth["Cp"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])
    adiabat_smooth["Cv_SI"] = adiabat_smooth["Cv"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])

    depths_km = adiabat["depths"] / 1e3

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
        ax.plot(adiabat[key], depths_km, color="0.75", linewidth=0.6)
        ax.plot(adiabat_smooth[key], depths_km, linewidth=1, color="C0")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.grid(alpha=0.2)
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)

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
