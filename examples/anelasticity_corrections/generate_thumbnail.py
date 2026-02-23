#!/usr/bin/env python3
"""Generate thumbnail for the anelasticity corrections example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gdrift

THUMBNAIL_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "thumbnails"


def generate():
    slb = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    test_depth = 500e3
    dense_temps = np.linspace(1200, 3500, 200)
    dense_depths = np.full_like(dense_temps, test_depth)
    vs_el = slb.temperature_to_vs(temperature=dense_temps, depth=dense_depths)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for qname in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        model = gdrift.CammaranoAnelasticityModel.from_q_profile(qname)
        corrected = gdrift.apply_anelastic_correction(slb, model)
        vs_an = corrected.temperature_to_vs(
            temperature=dense_temps, depth=dense_depths)
        reduction = (vs_el - vs_an) / vs_el * 100
        ax.plot(dense_temps, reduction, "-", label=f"Cam. {qname}",
                linewidth=1.2)

    for qname in ["Q1", "Q2"]:
        model = gdrift.GoesAnelasticityModel.from_q_profile(qname)
        corrected = gdrift.apply_anelastic_correction(slb, model)
        vs_an = corrected.temperature_to_vs(
            temperature=dense_temps, depth=dense_depths)
        reduction = (vs_el - vs_an) / vs_el * 100
        ax.plot(dense_temps, reduction, "--", label=f"Goes {qname}",
                linewidth=1.2)

    ax.set_xlabel("Temperature [K]", fontsize=9)
    ax.set_ylabel("Vs Reduction [%]", fontsize=9)
    ax.set_title(f"Velocity reduction at {test_depth / 1e3:.0f} km", fontsize=10)
    ax.legend(ncol=2, fontsize=6)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBNAIL_DIR / "anelasticity_corrections.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate()
