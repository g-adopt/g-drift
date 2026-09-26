#!/usr/bin/env python3
"""Convert LLNL-G3D-JPS from authoritative NetCDF4 to gdrift HDF5 point cloud.

Reads the original LLNL_G3D_JPS.nc (NetCDF4/HDF5) directly from LLNL, which
contains Vp, Vs, dlnVp, dlnVs on a 1-degree grid with 60 depth levels
(including discontinuity straddle levels).

Uses h5py (not netCDF4 library) since the file is HDF5 under the hood.
Resamples onto a 65,341-point Fibonacci sphere per depth level using IDW
interpolation (power=2, k=10 neighbours), consistent with convert_seismic_models.py.

The source stores each boundary as two levels about 0.1 km apart, one level
for each side of the boundary. At the boundaries from the upper crust down to
660 km, the source puts the level with the values of the LOWER unit at the
smaller depth. gdrift interpolates linearly in radius between the two levels
that bracket a query point, so this order makes the interpolation mix the
wrong units: for example, the lower-crust values at the Moho are blended with
the mantle down to 72.5 km. `fix_boundary_order` swaps the depths of the two
members of these pairs before the conversion. See its docstring for the pairs
and for the levels above 12.4 km, which stay as they are.

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

# Boundary pairs of the source whose two levels are in reversed order: the
# level with the lower unit's values is at the smaller depth. Each entry holds
# the two depths in km as the source prints them to four decimals, and the name
# of the boundary. The stored depths have more digits (12.487752885630353 km),
# so fix_boundary_order matches them with the tolerance below.
REVERSED_BOUNDARY_PAIRS_KM = [
    (12.4878, 12.5878, "upper crust / middle crust"),
    (18.6298, 18.7298, "middle crust / lower crust"),
    (26.3947, 26.4947, "Moho"),
    (415.6032, 415.7032, "410 km discontinuity"),
    (657.5472, 657.5496, "660 km discontinuity"),
]

# Matching tolerance for the pair depths, in km (1 m). The two levels at
# 660 km are 2.4 m apart, so a tolerance of 1 m still separates them.
PAIR_TOLERANCE_KM = 1e-3


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


def fix_boundary_order(depth_km, r_m, vp, lat):
    """Swap the depths of the reversed boundary pairs of the LLNL source.

    For each pair in REVERSED_BOUNDARY_PAIRS_KM the function finds the two
    source levels and swaps their depth and radius. The field values of each
    level do not change; only the depth assigned to them changes. After the
    swap, the level with the upper unit's values is the shallower member of
    the pair, so the mean Vp increases with depth across each boundary.

    Before a swap the function checks that the shallower member has the larger
    area-weighted mean Vp, which is the reversed order that the source has.
    If the check fails, the source is already in order (for example a
    corrected release of the file) and the function raises, so it can never
    reverse a correct file.

    The levels above 12.4 km (water, ice, three sediment units and the top of
    the upper crust) stay as they are. In the source they are a crust of
    laterally variable thickness put at fixed mean depths, so no order of
    them is physical, and a sort by velocity would put the ice level (Vp 3.81
    km/s, Vs 1.94 km/s) below the sediments.

    Parameters
    ----------
    depth_km : numpy.ndarray
        Depth of each source level in km, shape (num_layers,).
    r_m : numpy.ndarray
        Radius of each source level in m, shape (num_layers,).
    vp : numpy.ndarray
        P-wave speed in km/s, shape (num_layers, num_lat, num_lon).
    lat : numpy.ndarray
        Latitudes of the grid in degrees, shape (num_lat,). They give the
        cos(latitude) weights of the layer means.

    Returns
    -------
    depth_fixed, r_fixed : numpy.ndarray
        Copies of depth_km and r_m with the pairs swapped.

    Raises
    ------
    ValueError
        If a pair depth matches no level or more than one level, or if a pair
        is not in the reversed order described above.
    """
    depth_fixed = np.array(depth_km, dtype=float, copy=True)
    r_fixed = np.array(r_m, dtype=float, copy=True)

    # Area weights of the regular grid: cos(latitude), the same for every
    # longitude, broadcast over the (lat, lon) plane of one level.
    weights = np.cos(np.radians(lat))[:, np.newaxis] * np.ones(vp.shape[2])

    def level_index(depth):
        """Return the one source level within PAIR_TOLERANCE_KM of depth."""
        matches = np.flatnonzero(np.abs(depth_km - depth) < PAIR_TOLERANCE_KM)
        if matches.size != 1:
            raise ValueError(f"expected one level at {depth} km, found {matches.size}")
        return int(matches[0])

    def mean_vp(index):
        """Area-weighted mean Vp of one source level, in km/s."""
        return float(np.sum(weights * vp[index]) / np.sum(weights))

    for shallow_depth, deep_depth, name in REVERSED_BOUNDARY_PAIRS_KM:
        i_shallow = level_index(shallow_depth)
        i_deep = level_index(deep_depth)
        vp_shallow, vp_deep = mean_vp(i_shallow), mean_vp(i_deep)
        # The reversed order: the faster (lower) unit sits at the smaller depth.
        if not vp_shallow > vp_deep:
            raise ValueError(
                f"{name}: the level at {depth_km[i_shallow]:.4f} km has mean Vp "
                f"{vp_shallow:.3f} km/s and the level at {depth_km[i_deep]:.4f} km "
                f"{vp_deep:.3f} km/s. The pair is not reversed, so the source "
                "differs from the one this fix is for."
            )
        # Swap depth and radius. The radii of the source satisfy
        # r = 6371 km - depth, so swapping both keeps them consistent.
        depth_fixed[[i_shallow, i_deep]] = depth_fixed[[i_deep, i_shallow]]
        r_fixed[[i_shallow, i_deep]] = r_fixed[[i_deep, i_shallow]]
        print(
            f"  {name}: Vp {vp_deep:.3f} km/s now at {depth_fixed[i_deep]:.4f} km, "
            f"Vp {vp_shallow:.3f} km/s now at {depth_fixed[i_shallow]:.4f} km"
        )

    return depth_fixed, r_fixed


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

    # Put the members of the reversed boundary pairs in the right order.
    print("Fixing the order of the boundary pairs ...")
    src["depth"], src["r"] = fix_boundary_order(src["depth"], src["r"], src["Vp"], src["lat"])

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
            f"(siavash.ghelichkhan@anu.edu.au). "
            "The depths of the two levels of each boundary pair at 12.5, 18.7, 26.4, "
            "415.7 and 657.5 km are swapped relative to the source, which put the lower "
            "unit's level above the upper unit's level. The levels above 12.4 km are a "
            "crust of variable thickness at fixed mean depths and are not located physically."
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
