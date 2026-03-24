#!/usr/bin/env python3
"""Reinterpolate REVEAL tomography model using Wendland C2 kernel.

Reads the original REVEAL.nc (0.5-degree lat/lon grid) and resamples onto
Fibonacci sphere grids with depth-dependent resolution:

    0-500 km   : 260,000 pts/layer  (matches source 0.5-degree density)
    500-1500 km: 130,000 pts/layer
    1500-2880 km:  65,341 pts/layer  (standard g-drift density)

The KD-tree is built on 3D unit-sphere coordinates (not raw lat/lon) to
avoid polar distortion.  Interpolation is done per depth layer to prevent
vertical contamination between levels.

Output:
    gdrift/data-sia/3d_seismic_REVEAL.h5   (human-readable name)
    gdrift/data/<hash>.h5                   (runtime cache)
"""

import sys
from pathlib import Path

import hashlib
import numpy as np
from scipy.io import netcdf_file
from scipy.spatial import cKDTree
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdrift.utility import fibonacci_sphere, cartesian_to_geodetic
from gdrift.constants import R_earth
from gdrift.datasetnames import hash_name

# ── Configuration ──────────────────────────────────────────────────────────
SOURCE_PATH = Path("/Users/sghelichkhani/Downloads/REVEAL.nc")
OUTPUT_SIA = Path("gdrift/data-sia/3d_seismic_REVEAL.h5")

# Uniform resolution matching all other g-drift seismic models.
# Using a single layer size ensures _detect_layer_structure() in
# EarthModel3D recognises the layered layout and uses the clean
# lateral+radial interpolation path instead of a raw 3D KD-tree
# (which mixes depth layers and produces noisy results).
RESOLUTION_TIERS = [
    (9999, 65341),    # standard g-drift density for all depths
]

NUM_NEIGHBOURS = 20              # enough neighbours to fill Wendland support
SUPPORT_RADIUS_DEG = 1.5         # Wendland support radius in degrees
VELOCITY_FIELDS = {"vsv", "vsh", "vpv"}  # fields to scale km/s -> m/s


def latlon_to_unit_sphere(lat_deg, lon_deg):
    """Convert lat/lon (degrees) to 3D Cartesian on the unit sphere."""
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def wendland_c2_weights(dists, support_radius):
    """Wendland C2 kernel: (1-q)^4 * (4q+1) for q = d/h <= 1."""
    q = dists / support_radius
    weights = np.zeros_like(q)
    mask = q <= 1.0
    weights[mask] = (1 - q[mask]) ** 4 * (4 * q[mask] + 1)
    return weights


def get_tier(depth_km):
    """Return the number of Fibonacci points for a given depth."""
    for threshold, n_pts in RESOLUTION_TIERS:
        if depth_km <= threshold:
            return n_pts
    return RESOLUTION_TIERS[-1][1]


def main():
    # ── Read source data ───────────────────────────────────────────────────
    print(f"Reading {SOURCE_PATH} ...")
    f = netcdf_file(str(SOURCE_PATH), "r", mmap=False)
    depths_km = f.variables["depth"].data.copy()
    src_lats = f.variables["latitude"].data.copy()
    src_lons = f.variables["longitude"].data.copy()

    field_data = {}
    for name in ["vsv", "vsh", "vpv", "rho"]:
        field_data[name] = f.variables[name].data.copy()
    f.close()

    print(f"  {len(depths_km)} depths, {len(src_lats)}x{len(src_lons)} "
          f"lat/lon grid ({len(src_lats) * len(src_lons)} pts/layer)")

    # ── Build source 2D grid on unit sphere ────────────────────────────────
    lons_2d, lats_2d = np.meshgrid(src_lons, src_lats, indexing="xy")
    source_xyz = latlon_to_unit_sphere(lats_2d.ravel(), lons_2d.ravel())

    print("Building KD-tree on unit sphere ...")
    tree = cKDTree(source_xyz)

    # Convert support radius to chord distance on unit sphere
    support_chord = 2.0 * np.sin(np.radians(SUPPORT_RADIUS_DEG) / 2.0)
    print(f"  Wendland C2 support: {SUPPORT_RADIUS_DEG}° "
          f"-> chord = {support_chord:.6f}")

    # ── Pre-compute Fibonacci grids and KD-tree queries per tier ───────────
    unique_sizes = sorted(set(get_tier(d) for d in depths_km), reverse=True)
    tier_cache = {}  # n_pts -> (fib_coords, weights, weight_sum, close_mask, indices)

    for n_pts in unique_sizes:
        print(f"Preparing Fibonacci tier: {n_pts:,} pts/layer ...")
        fib = fibonacci_sphere(n_pts)
        fib_lats, fib_lons, _ = cartesian_to_geodetic(*fib.T)
        target_xyz = latlon_to_unit_sphere(fib_lats, fib_lons)

        dists, indices = tree.query(target_xyz, k=NUM_NEIGHBOURS)
        weights = wendland_c2_weights(dists, support_chord)
        weights = np.maximum(weights, 1e-12)
        weight_sum = np.sum(weights, axis=1)
        close_mask = dists[:, 0] < 1e-10

        zero_weight = weight_sum < 1e-10
        if np.any(zero_weight):
            print(f"  WARNING: {np.sum(zero_weight)} points with zero weight")

        tier_cache[n_pts] = (fib, weights, weight_sum, close_mask, indices)

    # ── Report depth-tier mapping ──────────────────────────────────────────
    total_pts = sum(get_tier(d) for d in depths_km)
    print(f"\nDepth-resolution plan ({len(depths_km)} layers, {total_pts:,} total points):")
    for threshold, n_pts in RESOLUTION_TIERS:
        layer_count = sum(1 for d in depths_km if get_tier(d) == n_pts)
        if layer_count > 0:
            d_in_tier = [d for d in depths_km if get_tier(d) == n_pts]
            print(f"  {d_in_tier[0]:.0f}-{d_in_tier[-1]:.0f} km: "
                  f"{layer_count} layers x {n_pts:,} pts = {layer_count * n_pts:,}")

    # ── Build coordinates and interpolate per depth layer ──────────────────
    all_coords = []
    all_fields = {name: [] for name in field_data}

    for i, depth_km in enumerate(depths_km):
        n_pts = get_tier(depth_km)
        fib, weights, weight_sum, close_mask, indices = tier_cache[n_pts]

        # Cartesian coordinates at this depth
        layer_coords = fib * (R_earth - depth_km * 1e3)
        all_coords.append(layer_coords)

        # Interpolate each field
        for name, data_3d in field_data.items():
            layer = data_3d[i].ravel()
            vals_at_nb = layer[indices]
            weighted = np.sum(vals_at_nb * weights, axis=1) / weight_sum
            weighted[close_mask] = layer[indices[close_mask, 0]]
            all_fields[name].append(weighted)

    # ── Concatenate ────────────────────────────────────────────────────────
    print("\nConcatenating arrays ...")
    coordinates = np.vstack(all_coords)
    print(f"  coordinates: {coordinates.shape}")

    results = {}
    for name in field_data:
        arr = np.concatenate(all_fields[name])
        if name in VELOCITY_FIELDS:
            arr *= 1e3
            units = "m/s"
        else:
            units = "kg/m3"
        results[name] = arr
        print(f"  {name}: min={np.nanmin(arr):.4f}  "
              f"max={np.nanmax(arr):.4f}  mean={np.nanmean(arr):.4f}  ({units})")

    # ── Write HDF5 ─────────────────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_SIA} ...")
    OUTPUT_SIA.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_SIA, "w") as hf:
        hf.create_dataset("coordinates", data=coordinates)
        for name, arr in results.items():
            hf.create_dataset(name, data=arr)
        hf.attrs["source"] = "REVEAL tomography model (Thrastarson et al.)"
        hf.attrs["comment"] = (
            "Resampled to depth-dependent Fibonacci sphere "
            "(260k/130k/65k pts per layer for shallow/mid/deep mantle, "
            f"{NUM_NEIGHBOURS} neighbours, Wendland C2 kernel, "
            f"support={SUPPORT_RADIUS_DEG} deg) by Sia Ghelichkhan "
            "(siavash.ghelichkhan@anu.edu.au)"
        )
        hf.attrs["velocity_units"] = "m/s"
        hf.attrs["density_units"] = "kg/m3"

    # Compute hash and copy to runtime cache
    sha256 = hashlib.sha256()
    with open(OUTPUT_SIA, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    sha_hex = sha256.hexdigest()

    hashed_name = hash_name("3d_seismic_REVEAL")
    output_cache = Path(f"gdrift/data/{hashed_name}.h5")
    output_cache.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(OUTPUT_SIA, output_cache)
    print(f"Copied to {output_cache}")

    print(f"\nSHA256: {sha_hex}")
    print(f"Update datasets.json with this hash for 3d_seismic_REVEAL")
    print("Done.")


if __name__ == "__main__":
    main()
