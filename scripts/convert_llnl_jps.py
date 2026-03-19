#!/usr/bin/env python3
"""Convert LLNL-G3D-JPS from authoritative NetCDF4 to gdrift HDF5 point cloud.

Reads the original LLNL_G3D_JPS.nc (NetCDF4/HDF5) directly from LLNL, which
contains Vp, Vs, dlnVp, dlnVs on a 1-degree grid with 60 depth levels
(including discontinuity straddle levels).

Uses h5py (not netCDF4 library) since the file is HDF5 under the hood.
Resamples onto a 65,341-point Fibonacci sphere per depth level using IDW
interpolation (power=2, k=10 neighbours), consistent with convert_seismic_models.py.

Usage:
    python scripts/convert_llnl_jps.py --source LLNL_G3D_JPS.nc
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdrift.utility import fibonacci_sphere, cartesian_to_geodetic
from gdrift.datasetnames import hash_name

DATASET_NAME = "3d_seismic_LLNL-G3D-JPS"
NUM_SAMPLES = 65341
NUM_NEIGHBOURS = 10
FAR_THRESHOLD = 5.0  # degrees


def read_source(filepath):
    """Read the LLNL NetCDF4 file via h5py."""
    f = h5py.File(filepath, "r")
    data = {
        "lat": f["lat"][:],
        "lon": f["lon"][:],
        "depth": f["depth"][:],
        "r": f["r"][:],
        "Vp": f["Vp"][:],
        "Vs": f["Vs"][:],
        "dlnVp_percent": f["dlnVp_percent"][:],
        "dlnVs_percent": f["dlnVs_percent"][:],
    }
    attrs = dict(f.attrs)
    f.close()
    return data, attrs


def interpolate_field(field_3d, tree, dists, indices, weights, weight_sum,
                      close_points, far_points, num_layers, num_points):
    """Interpolate a 3D field (depth, lat, lon) onto Fibonacci points."""
    result = np.empty((num_layers, num_points))

    for layer in range(num_layers):
        flat = field_3d[layer].ravel()
        vals_at_nb = flat[indices]
        weighted = np.einsum("pk,pk->p", vals_at_nb, weights) / weight_sum
        weighted[close_points] = flat[indices[close_points, 0]]
        weighted[far_points] = np.nan
        result[layer] = weighted

    return result.ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("LLNL_G3D_JPS.nc"),
                        help="Path to LLNL_G3D_JPS.nc")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")

    # Read source
    print(f"Reading {args.source} ...")
    src, attrs = read_source(args.source)
    num_layers = len(src["depth"])
    print(f"  {num_layers} depths, {len(src['lat'])}x{len(src['lon'])} grid")
    print(f"  Depth range: {src['depth'].min():.2f} - {src['depth'].max():.2f} km")

    # Generate Fibonacci sphere
    print(f"Generating Fibonacci sphere ({NUM_SAMPLES} points) ...")
    fib_coords = fibonacci_sphere(NUM_SAMPLES)
    fib_lats, fib_lons, _ = cartesian_to_geodetic(*fib_coords.T)

    # Build KD-tree on flat lon/lat
    lons_grid, lats_grid = np.meshgrid(src["lon"], src["lat"], indexing="xy")
    tree = cKDTree(np.column_stack((lons_grid.ravel(), lats_grid.ravel())))
    dists, indices = tree.query(
        np.column_stack((fib_lons, fib_lats)), k=NUM_NEIGHBOURS)

    close_points = dists[:, 0] < 1e-10
    far_points = dists[:, 0] > FAR_THRESHOLD

    if np.any(far_points):
        print(f"  WARNING: {np.sum(far_points)} points beyond {FAR_THRESHOLD} deg threshold")

    # 1/d^2 weights
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / (dists ** 2)
        weight_sum = np.sum(weights, axis=1)

    # Build Cartesian coordinates using r directly (metres from Earth centre)
    print("Building coordinates from radius array ...")
    coords_list = [fib_coords * src["r"][i] for i in range(num_layers)]
    coordinates = np.vstack(coords_list)
    print(f"  coordinates shape: {coordinates.shape}")

    # Interpolate and convert each field
    field_map = [
        ("Vp",              "vp",  1e3),    # km/s -> m/s
        ("Vs",              "vs",  1e3),    # km/s -> m/s
        ("dlnVp_percent",   "dvp", 1e-2),   # percent -> fractional
        ("dlnVs_percent",   "dvs", 1e-2),   # percent -> fractional
    ]

    data_to_write = {"coordinates": coordinates}
    fields_written = []

    for src_name, tgt_name, scale in field_map:
        print(f"Interpolating {src_name} -> {tgt_name} ...")
        interpolated = interpolate_field(
            src[src_name], tree, dists, indices, weights, weight_sum,
            close_points, far_points, num_layers, NUM_SAMPLES)
        interpolated *= scale
        data_to_write[tgt_name] = interpolated
        fields_written.append(tgt_name)

        n_nan = np.sum(np.isnan(interpolated))
        print(f"  {tgt_name}: min={np.nanmin(interpolated):.4f}  "
              f"max={np.nanmax(interpolated):.4f}  "
              f"mean={np.nanmean(interpolated):.4f}  NaN={n_nan}")

    # Write HDF5
    output_sia = Path("gdrift/data-sia") / f"{DATASET_NAME}.h5"
    output_sia.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {output_sia} ...")

    with h5py.File(output_sia, "w") as f:
        for key, value in data_to_write.items():
            f.create_dataset(key, data=value)
        f.attrs["velocity_units"] = "m/s"
        f.attrs["perturbation_units"] = "fractional"
        f.attrs["source"] = (
            "Simmons, N. A., Myers, S. C., Johannesson, G., Matzel, E., & Grand, S. P. (2015). "
            "Evidence for long-lived subduction of an ancient tectonic plate beneath the southern "
            "Indian Ocean. Geophysical Research Letters, 42, 9270-9278."
        )
        f.attrs["doi"] = "10.1002/2015GL066237"
        f.attrs["comment"] = (
            f"Converted from authoritative LLNL NetCDF4 source. "
            f"Resampled to Fibonacci sphere ({NUM_SAMPLES} points per layer, "
            f"{NUM_NEIGHBOURS} neighbours, IDW power=2) by Sia Ghelichkhan "
            f"(siavash.ghelichkhan@anu.edu.au)"
        )

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(output_sia, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    sha_hex = sha256.hexdigest()

    # Copy to runtime cache
    hashed = hash_name(DATASET_NAME)
    output_cache = Path(f"gdrift/data/{hashed}.h5")
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_sia, output_cache)
    print(f"Copied to {output_cache}")

    print(f"\nSHA256: {sha_hex}")
    print(f"Fields: {sorted(fields_written)}")
    print(f"Total points: {coordinates.shape[0]}")
    print(f"\nUpdate datasets.json: set sha256 to '{sha_hex}' "
          f"and fields to {sorted(fields_written)}")
    print("Done.")


if __name__ == "__main__":
    main()
