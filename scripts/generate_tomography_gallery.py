#!/usr/bin/env python3
"""Generate tomography gallery images and markdown for the data catalog page.

For each of the 47 seismic tomography models available in gdrift, this script:
1. Classifies the model as global or regional based on lat/lon extent
2. Groups models by field category (dvs, dvp, vs, vp, vsv, etc.)
3. Generates a wedge cross-section plot (global) or depth-slice map (regional)
4. Outputs individual PNGs and a gallery markdown file for mkdocs inclusion

Usage:
    MPLBACKEND=Agg python scripts/generate_tomography_gallery.py

Outputs:
    docs/assets/images/tomography/*.png
    docs/tomography-gallery-generated.md
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

# Ensure gdrift is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdrift  # noqa: E402
from gdrift.constants import R_earth, R_cmb  # noqa: E402

# Output paths
IMAGE_DIR = Path("docs/assets/images/tomography")
GALLERY_MD = Path("docs/tomography-gallery-generated.md")

# Plotting defaults
CMAP = "RdBu"
FIG_SIZE = (3.5, 3.5)
DPI = 120

# Cross-section sampling
N_LAT = 180  # latitude samples per side of the great circle
N_DEPTH = 60  # depth samples

# Cross-section latitude bounds (wedge shape)
LAT_NORTH = 20.0   # northern bound (degrees)
LAT_SOUTH = -70.0   # southern bound (degrees)

# Depth-slice sampling for regional models
REGIONAL_LAT_RES = 1.0  # degrees
REGIONAL_LON_RES = 1.0  # degrees

# Fields of interest (ordered for display)
FIELD_ORDER = ["dvs", "dvp", "vs", "vp", "vsv", "vsh", "vpv", "vph"]


def classify_model(model):
    """Classify a seismic model as global or regional.

    Uses only coordinates with valid (non-NaN) data to determine extent,
    since some models (e.g. MITS-18) have a global coordinate grid but
    valid data only in a regional subset.

    Returns (is_global, lat_range, lon_range, depth_range).
    """
    coords = model.coordinates
    lat, lon, depth = gdrift.cartesian_to_geodetic(
        coords[:, 0], coords[:, 1], coords[:, 2]
    )

    # Build mask of points that have valid data in at least one field
    valid = np.zeros(len(lat), dtype=bool)
    for field, values in model.available_fields.items():
        if field == "coordinates":
            continue
        valid |= ~np.isnan(values)

    if valid.any():
        lat, lon, depth = lat[valid], lon[valid], depth[valid]

    lat_range = (float(lat.min()), float(lat.max()))
    lon_range = (float(lon.min()), float(lon.max()))
    depth_range = (float(depth.min()), float(depth.max()))

    is_global = (
        lat_range[0] < -80 and lat_range[1] > 80
        and lon_range[0] < -170 and lon_range[1] > 170
    )
    return is_global, lat_range, lon_range, depth_range


def make_cross_section_coords(lat_north=LAT_NORTH, lat_south=LAT_SOUTH):
    """Build query coordinates for a wedge cross-section along 0/180 longitude.

    The wedge spans from lat_north to lat_south on both sides of the
    great circle.  In polar coordinates theta=0 is placed at the top
    (lat_north) using theta_zero_location("N").  Positive theta goes
    clockwise (lon=0 side); negative theta goes counter-clockwise
    (lon=180 side).

    Returns (theta_grid, r_grid, query_coords) where query_coords is Nx3.
    """
    half_span = np.radians(lat_north - lat_south)
    theta = np.linspace(-half_span, half_span, 2 * N_LAT)
    depths = np.linspace(0, R_earth - R_cmb, N_DEPTH)
    theta_grid, depth_grid = np.meshgrid(theta, depths)
    r_grid = R_earth - depth_grid

    # theta > 0 -> lon=0 (right side), theta < 0 -> lon=180 (left side)
    # latitude decreases symmetrically from lat_north at theta=0
    lats = lat_north - np.degrees(np.abs(theta_grid))
    lons = np.where(theta_grid >= 0, 0.0, 180.0)

    query_coords = gdrift.geodetic_to_cartesian(
        lats.ravel(), lons.ravel(), depth_grid.ravel()
    )
    return theta_grid, r_grid, query_coords


def make_regional_coords(lat_range, lon_range, depth_range):
    """Build query coordinates for a depth-slice at the mid-depth of a regional model.

    Returns (lat_grid, lon_grid, mid_depth, query_coords).
    """
    mid_depth = (depth_range[0] + depth_range[1]) / 2.0
    lats = np.arange(lat_range[0], lat_range[1] + REGIONAL_LAT_RES, REGIONAL_LAT_RES)
    lons = np.arange(lon_range[0], lon_range[1] + REGIONAL_LON_RES, REGIONAL_LON_RES)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    depth_arr = np.full_like(lat_grid, mid_depth)
    query_coords = gdrift.geodetic_to_cartesian(
        lat_grid.ravel(), lon_grid.ravel(), depth_arr.ravel()
    )
    return lat_grid, lon_grid, mid_depth, query_coords


def _format_field_label(field):
    """Format a field name for display, e.g. 'dvs' -> 'dVs', 'vsh' -> 'Vsh'."""
    labels = {
        "dvs": "dVs", "dvp": "dVp",
        "vs": "Vs", "vp": "Vp",
        "vsv": "Vsv", "vsh": "Vsh",
        "vpv": "Vpv", "vph": "Vph",
    }
    return labels.get(field, field.capitalize())


def _format_lat(deg):
    """Format a latitude value as e.g. '20\u00b0N' or '70\u00b0S'."""
    if deg > 0:
        return f"{deg:.0f}\u00b0N"
    elif deg < 0:
        return f"{abs(deg):.0f}\u00b0S"
    return "Eq"


def plot_global_cross_section(model_name, field, theta_grid, r_grid, values, out_path,
                               lat_north=LAT_NORTH, lat_south=LAT_SOUTH):
    """Plot a wedge cross-section in polar projection."""
    data = values.reshape(theta_grid.shape)
    half_span_deg = lat_north - lat_south

    fig, ax = plt.subplots(
        figsize=FIG_SIZE, subplot_kw={"projection": "polar"}, dpi=DPI
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax.pcolormesh(theta_grid, r_grid / 1e3, data, cmap=CMAP, norm=norm, shading="auto")

    ax.set_thetamin(-half_span_deg)
    ax.set_thetamax(half_span_deg)
    ax.set_rorigin(0)
    ax.set_rlim(R_cmb / 1e3, R_earth / 1e3)

    ax.set_yticks([])

    # Tick labels: top = lat_north, bottom endpoints = longitude labels
    ticks = [0, np.radians(half_span_deg), np.radians(-half_span_deg)]
    labels = [_format_lat(lat_north), "0\u00b0", "180\u00b0"]

    # Add equator ticks if the latitude range spans the equator
    if lat_south < 0 < lat_north:
        eq_theta = np.radians(lat_north)
        ticks.extend([eq_theta, -eq_theta])
        labels.extend(["Eq", "Eq"])

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    ax.tick_params(axis="x", pad=1)
    ax.set_title(f"{model_name}\n{field}", fontsize=8, pad=8)
    ax.grid(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regional_slice(model_name, field, lat_grid, lon_grid, values, mid_depth, out_path):
    """Plot a depth-slice map for a regional model."""
    data = values.reshape(lat_grid.shape)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        lat_center = (lat_grid.min() + lat_grid.max()) / 2
        lon_center = (lon_grid.min() + lon_grid.max()) / 2

        # Use AlbersEqualArea for small regions, Mollweide for large
        lat_span = lat_grid.max() - lat_grid.min()
        lon_span = lon_grid.max() - lon_grid.min()
        if lat_span < 120 and lon_span < 180:
            projection = ccrs.AlbersEqualArea(
                central_longitude=lon_center, central_latitude=lat_center
            )
        else:
            projection = ccrs.Mollweide(central_longitude=lon_center)

        fig, ax = plt.subplots(
            figsize=FIG_SIZE, subplot_kw={"projection": projection}, dpi=DPI
        )

        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        ax.pcolormesh(
            lon_grid, lat_grid, data,
            cmap=CMAP, norm=norm, shading="auto",
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="k")
        ax.set_extent(
            [lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
            crs=ccrs.PlateCarree(),
        )
        ax.set_title(
            f"{model_name}\n{field} @ {mid_depth / 1e3:.0f} km", fontsize=8, pad=6
        )

    except ImportError:
        # Fallback without cartopy
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        ax.pcolormesh(lon_grid, lat_grid, data, cmap=CMAP, norm=norm, shading="auto")
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)
        ax.set_title(
            f"{model_name}\n{field} @ {mid_depth / 1e3:.0f} km", fontsize=8, pad=6
        )
        ax.set_aspect("equal")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_reference_map(lat_north=LAT_NORTH, lat_south=LAT_SOUTH):
    """Generate a reference map showing the cross-section path and latitude bounds."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig, ax = plt.subplots(
            figsize=(6, 3),
            subplot_kw={"projection": ccrs.Robinson()},
            dpi=DPI,
        )
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#e8e8e8")

        # Draw the great circle within the latitude bounds
        lats = np.linspace(lat_south, lat_north, 200)
        ax.plot(
            np.zeros_like(lats), lats,
            "r-", linewidth=2, transform=ccrs.PlateCarree(),
        )
        ax.plot(
            np.full_like(lats, 180), lats,
            "r-", linewidth=2, transform=ccrs.PlateCarree(),
        )

        # Draw latitude bound lines
        lons_line = np.linspace(-180, 180, 360)
        for lat_bound in [lat_north, lat_south]:
            ax.plot(
                lons_line, np.full_like(lons_line, lat_bound),
                "r--", linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(),
            )

        ax.set_title(
            f"Cross-section path (0\u00b0/180\u00b0, "
            f"{_format_lat(lat_north)} to {_format_lat(lat_south)})",
            fontsize=10,
        )

    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=DPI)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.axvline(0, color="r", linewidth=2)
        ax.axvline(180, color="r", linewidth=2)
        ax.axhline(lat_north, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(lat_south, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"Cross-section path (0\u00b0/180\u00b0, "
            f"{_format_lat(lat_north)} to {_format_lat(lat_south)})",
            fontsize=10,
        )
        ax.set_aspect("equal")

    out_path = IMAGE_DIR / "reference_cross_section.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def sanitize_filename(name):
    """Turn a model name into a safe filename component."""
    return name.replace("+", "plus").replace("/", "_").replace(" ", "_")


def generate_gallery():
    """Main entry point: generate all gallery images and the markdown file."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating reference cross-section map...")
    ref_map_path = generate_reference_map()
    print(f"  -> {ref_map_path}")

    # Pre-compute shared cross-section coordinates
    print("Building cross-section query grid...")
    theta_grid, r_grid, xsec_coords = make_cross_section_coords()
    print(f"  {xsec_coords.shape[0]} query points")

    # Collect results: {field: [(model_name, is_global, png_path), ...]}
    gallery = {}
    all_models = gdrift.AVAILABLE_SEISMIC_MODELS

    for i, model_name in enumerate(all_models):
        print(f"[{i + 1}/{len(all_models)}] Loading {model_name}...")
        try:
            model = gdrift.SeismicModel(model_name)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        is_global, lat_range, lon_range, depth_range = classify_model(model)
        tag = "global" if is_global else "regional"
        print(f"  {tag}, fields: {list(model.available_fields.keys())}")

        # Build regional coords once per model (if needed)
        regional_data = None
        if not is_global:
            regional_data = make_regional_coords(lat_range, lon_range, depth_range)

        for field in model.available_fields.keys():
            if field == "coordinates":
                continue
            safe_name = sanitize_filename(model_name)
            png_name = f"{safe_name}_{field}.png"
            out_path = IMAGE_DIR / png_name

            print(f"  Plotting {field}...")
            try:
                if is_global:
                    values = model.at(field, xsec_coords)
                    plot_global_cross_section(
                        model_name, field, theta_grid, r_grid, values, out_path
                    )
                else:
                    lat_grid, lon_grid, mid_depth, reg_coords = regional_data
                    values = model.at(field, reg_coords)
                    plot_regional_slice(
                        model_name, field, lat_grid, lon_grid, values, mid_depth, out_path
                    )
            except Exception as e:
                print(f"    ERROR plotting {model_name}/{field}: {e}")
                continue

            gallery.setdefault(field, []).append((model_name, is_global, png_name))
            print(f"    -> {out_path}")

    # Write the gallery markdown
    print(f"\nWriting gallery markdown to {GALLERY_MD}...")
    write_gallery_markdown(gallery)
    print("Done.")


def write_gallery_markdown(gallery):
    """Write the tomography-gallery-generated.md file.

    Image paths use ``../assets/`` because this file is snippet-included
    into ``data-catalog.md``, which MkDocs renders at
    ``data-catalog/index.html``.  Raw HTML ``<img src>`` attributes are
    not rewritten by MkDocs, so we need the ``../`` prefix to resolve
    correctly from the output directory.
    """
    img_prefix = "../assets/images/tomography"

    lines = []
    lines.append("## Tomography Model Gallery\n")
    lines.append(
        f"Cross-sections are taken along the 0\u00b0/180\u00b0 longitude great circle "
        f"between {_format_lat(LAT_NORTH)} and {_format_lat(LAT_SOUTH)} "
        f"(see reference map below). Regional models show a depth slice at the "
        f"model's mid-depth.\n"
    )
    lines.append(
        f'![Reference cross-section path]({img_prefix}/reference_cross_section.png)'
        '{: style="max-width:500px; display:block; margin:0 auto 1.5rem auto;" }\n'
    )

    for field in FIELD_ORDER:
        if field not in gallery:
            continue
        entries = gallery[field]
        field_label = _format_field_label(field)
        lines.append(f"### {field_label} Models\n")
        lines.append('<div class="tomography-grid">\n')

        for model_name, is_global, png_name in sorted(entries, key=lambda x: x[0].lower()):
            tag = "Global" if is_global else "Regional"
            lines.append(f'<div class="tomography-card">')
            lines.append(
                f'<img src="{img_prefix}/{png_name}" '
                f'alt="{model_name} {field}" loading="lazy">'
            )
            lines.append(f'<span class="tomography-label">{model_name}</span>')
            lines.append(f'<span class="tomography-tag">{tag}</span>')
            lines.append("</div>\n")

        lines.append("</div>\n")

    # Any fields not in FIELD_ORDER
    for field in sorted(gallery.keys()):
        if field in FIELD_ORDER:
            continue
        entries = gallery[field]
        lines.append(f"### {field} Models\n")
        lines.append('<div class="tomography-grid">\n')

        for model_name, is_global, png_name in sorted(entries, key=lambda x: x[0].lower()):
            tag = "Global" if is_global else "Regional"
            lines.append(f'<div class="tomography-card">')
            lines.append(
                f'<img src="{img_prefix}/{png_name}" '
                f'alt="{model_name} {field}" loading="lazy">'
            )
            lines.append(f'<span class="tomography-label">{model_name}</span>')
            lines.append(f'<span class="tomography-tag">{tag}</span>')
            lines.append("</div>\n")

        lines.append("</div>\n")

    GALLERY_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    generate_gallery()
