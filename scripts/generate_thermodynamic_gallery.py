#!/usr/bin/env python3
"""Generate thermodynamic model gallery images and markdown for the data catalog page.

For each thermodynamic model in the gallery scope (the SLB_21 family plus the new
full-iron SLB_24 CFMASNaCr pyrolite model), this script renders the 2D lookup
tables as depth-versus-temperature colour fields, one multi-panel figure per
model. Each panel shows a single property (Vs, Vp, density, shear modulus, bulk
modulus) painted directly from the stored table, so phase-transition steps are
visible without any interpolation smoothing.

Usage:
    MPLBACKEND=Agg python scripts/generate_thermodynamic_gallery.py

Outputs:
    docs/assets/images/thermodynamic/*.png
    docs/thermodynamic-gallery-generated.md
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

# Ensure gdrift is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdrift  # noqa: E402
from gdrift.datasetnames import DATASET_REGISTRY, DatasetType  # noqa: E402

# Output paths
IMAGE_DIR = Path("docs/assets/images/thermodynamic")
GALLERY_MD = Path("docs/thermodynamic-gallery-generated.md")

# Plotting defaults
CMAP = "RdBu"          # low values red, high values blue (mimics the reference panel)
DPI = 120
PANEL_SIZE = (2.8, 3.6)  # per-panel (width, height) in inches
PERCENTILE_CLIP = (2, 98)  # colour-scale clip to suppress outliers

# The featured new model (SLB 2024 "role of iron", full CFMASNaCr system from HeFESTo)
FEATURED_MODEL = "SLB_24_pyroliteCFMASNaCr"

# Properties to render per model: (table_key, label, unit, scale-to-display)
VARIABLES = [
    ("vs", r"$V_s$", "km/s", 1e-3),
    ("vp", r"$V_p$", "km/s", 1e-3),
    ("rho", r"$\rho$", "g/cm$^3$", 1e-3),
    ("shear_mod", r"Shear modulus $\mu$", "GPa", 1e-9),
    ("bulk_mod", r"Bulk modulus $K$", "GPa", 1e-9),
]


def select_models():
    """Return the ordered list of (name, model, composition, chem) tuples to render.

    Scope: every SLB_21 thermodynamic model plus the featured SLB_24 full-iron
    CFMASNaCr pyrolite model, all derived from the manifest so the gallery tracks
    whatever ships in datasets.json.
    """
    thermo = DATASET_REGISTRY.filter_by_type(DatasetType.THERMODYNAMIC_MODEL)
    by_name = {d.name: d for d in thermo}

    selected = []
    # Featured model first, if present
    if FEATURED_MODEL in by_name:
        selected.append(by_name[FEATURED_MODEL])
    # Then the SLB_21 family, sorted
    for d in sorted(thermo, key=lambda d: d.name):
        if d.name.startswith("SLB_21_"):
            selected.append(d)

    out = []
    for d in selected:
        model = "_".join(d.name.split("_")[:2])          # e.g. SLB_21
        composition = d.name.split("_", 2)[2]            # e.g. pyroliteCFMAS
        out.append((d.name, model, composition, d))
    return out


def split_composition(composition, dataset):
    """Return (composition_label, chemical_system) for a model.

    Prefers the manifest's explicit composition/chemical_system fields, falling
    back to splitting the raw composition string (e.g. 'pyroliteCFMAS').
    """
    comp = getattr(dataset, "composition", None)
    chem = getattr(dataset, "chemical_system", None)
    if comp and chem:
        return comp, chem
    # Fallback: the chemical system is the trailing all-caps run
    for i, ch in enumerate(composition):
        if ch.isupper():
            return composition[:i], composition[i:]
    return composition, ""


def plot_model(name, model, composition, dataset, out_path):
    """Render a multi-panel depth-temperature figure for one thermodynamic model."""
    tm = gdrift.ThermodynamicModel(model, composition)

    n = len(VARIABLES)
    fig, axes = plt.subplots(
        1, n,
        figsize=(PANEL_SIZE[0] * n, PANEL_SIZE[1]),
        dpi=DPI,
        constrained_layout=True,
    )

    for ax, (key, label, unit, scale) in zip(np.atleast_1d(axes), VARIABLES):
        table = tm._get_table(key)
        depth_km = table.get_x() / 1e3        # m -> km
        temp = table.get_y()                  # K
        vals = table.get_vals() * scale       # display units, shape (depth, temp)

        finite = vals[np.isfinite(vals)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, PERCENTILE_CLIP)
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

        pcm = ax.pcolormesh(temp, depth_km, vals, cmap=CMAP, norm=norm, shading="auto")

        ax.set_ylim(depth_km.max(), depth_km.min())  # depth increases downward
        ax.set_xlabel("Temperature [K]", fontsize=8)
        ax.set_title(f"{label} [{unit}]", fontsize=9)
        ax.tick_params(labelsize=7)

        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.02, fraction=0.05)
        cbar.ax.tick_params(labelsize=6)

    np.atleast_1d(axes)[0].set_ylabel("Depth [km]", fontsize=8)

    comp_label, chem = split_composition(composition, dataset)
    fig.suptitle(f"{name}   ({model} · {comp_label} · {chem})", fontsize=11)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_gallery_markdown(entries):
    """Write docs/thermodynamic-gallery-generated.md.

    Markdown image references would be rewritten by MkDocs, but we use raw HTML
    ``<img>`` cards (consistent with the tomography gallery) so paths need the
    ``../assets/…`` prefix to resolve from the ``data-catalog/index.html`` output.
    """
    html_img = "../assets/images/thermodynamic"

    lines = []
    lines.append("## Thermodynamic Model Gallery\n")
    lines.append(
        "Each panel shows a 2D lookup table painted directly over its depth "
        "(vertical) and temperature (horizontal) grid, so the phase-transition "
        "steps in the mantle are visible without any interpolation. Properties "
        "shown are shear velocity, compressional velocity, density, and the shear "
        "and bulk moduli. Tables are computed with the Stixrude & "
        "Lithgow-Bertelloni databases.\n"
    )

    # Featured model gets its own callout
    featured = [e for e in entries if e[0] == FEATURED_MODEL]
    rest = [e for e in entries if e[0] != FEATURED_MODEL]

    if featured:
        lines.append("### SLB_24 — full-iron CFMASNaCr pyrolite\n")
        lines.append(
            "The complete chemical system (CaO–FeO–MgO–Al₂O₃–SiO₂ plus "
            "Na₂O and Cr₂O₃) from Stixrude & Lithgow-Bertelloni (2024), "
            "*Thermodynamics of mantle minerals III: The role of iron* "
            "([DOI: 10.1093/gji/ggae126](https://doi.org/10.1093/gji/ggae126)).\n"
        )
        lines.append('<div class="thermo-grid">\n')
        for name, png_name, subtitle in featured:
            lines.append('<div class="tomography-card">')
            lines.append(f'<img src="{html_img}/{png_name}" alt="{name}" loading="lazy">')
            lines.append(f'<span class="tomography-label">{name}</span>')
            lines.append(f'<span class="tomography-tag">{subtitle}</span>')
            lines.append("</div>\n")
        lines.append("</div>\n")

    if rest:
        lines.append("### SLB_21 family\n")
        lines.append('<div class="thermo-grid">\n')
        for name, png_name, subtitle in rest:
            lines.append('<div class="tomography-card">')
            lines.append(f'<img src="{html_img}/{png_name}" alt="{name}" loading="lazy">')
            lines.append(f'<span class="tomography-label">{name}</span>')
            lines.append(f'<span class="tomography-tag">{subtitle}</span>')
            lines.append("</div>\n")
        lines.append("</div>\n")

    GALLERY_MD.write_text("\n".join(lines))


def generate_gallery():
    """Main entry point: render all gallery images and write the markdown."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    models = select_models()
    print(f"Rendering {len(models)} thermodynamic models...")

    entries = []
    for i, (name, model, composition, dataset) in enumerate(models):
        png_name = f"{name}.png"
        out_path = IMAGE_DIR / png_name
        print(f"[{i + 1}/{len(models)}] {name} -> {out_path}")
        try:
            plot_model(name, model, composition, dataset, out_path)
        except Exception as e:
            print(f"  ERROR rendering {name}: {e}")
            continue
        comp_label, chem = split_composition(composition, dataset)
        subtitle = f"{model} · {comp_label} · {chem}"
        entries.append((name, png_name, subtitle))

    print(f"\nWriting gallery markdown to {GALLERY_MD}...")
    write_gallery_markdown(entries)
    print("Done.")


if __name__ == "__main__":
    generate_gallery()
