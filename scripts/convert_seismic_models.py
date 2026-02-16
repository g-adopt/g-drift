#!/usr/bin/env python3
"""
Reconvert seismic tomography models from raw NetCDF to HDF5 point clouds.

Uses an explicit model registry (no filename parsing), 1/d^2 IDW interpolation,
consistent SI units (m/s for absolute velocities), and generates manifest entries
with a 'fields' key for datasets.json.

Supports two source formats:
  1. Mather collection (--source-dir): single NetCDF per field with 3D `v` array
  2. GRD collection (--grd-source-dir): directory of per-depth .grd files

Usage:
    # Old Mather collection
    python scripts/convert_seismic_models.py \
        --source-dir /path/to/seismic-tomography-models \
        --output-dir gdrift/data \
        --models S40RTS REVEAL

    # New GRD collection
    python scripts/convert_seismic_models.py \
        --grd-source-dir /path/to/tomography \
        --output-dir gdrift/data \
        --grd-models S40RTS CAM2016

    # Both at once
    python scripts/convert_seismic_models.py \
        --source-dir /path/to/mather \
        --grd-source-dir /path/to/tomography \
        --output-dir gdrift/data
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree

# ── gdrift imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdrift.utility import fibonacci_sphere, cartesian_to_geodetic
from gdrift.constants import R_earth
from gdrift.datasetnames import hash_name

# ── Canonical field names and which ones get scaled km/s -> m/s ─────────────
SCALE_TO_MS = {"vs", "vp", "vsh", "vsv", "vpv", "vph"}

# ── Model Registry ──────────────────────────────────────────────────────────
# Maps model name -> list of (filename, canonical_field_name).
# REVEAL is special: a single multi-variable file.
MODEL_REGISTRY = {
    "GAP": [
        ("GAP_P4_dvp.nc", "dvp"),
    ],
    "GyPSuM": [
        ("GyPSuM_vp.nc", "vp"),
        ("GyPSuM_vs.nc", "vs"),
    ],
    "HMSL-P06": [
        ("HMSL-P06_dvp.nc", "dvp"),
        ("HMSL-P06_vp.nc", "vp"),
    ],
    "HMSL-S06": [
        ("HMSL-S06_dvs.nc", "dvs"),
        ("HMSL-S06_vs.nc", "vs"),
    ],
    "LLNL-G3Dv3": [
        ("LLNL-G3Dv3_dvp.nc", "dvp"),
        ("LLNL-G3Dv3_vp.nc", "vp"),
    ],
    "MITP08": [
        ("MITP08_dvp.nc", "dvp"),
        ("MITP08_vp.nc", "vp"),
    ],
    "OJP": [
        ("OJP_P_dvp.nc", "dvp"),
    ],
    "REVEAL": "REVEAL",  # sentinel: handled by read_reveal_netcdf
    "S20RTS": [
        ("S20RTS_dvs.nc", "dvs"),
    ],
    "S40RTS": [
        ("S40RTS_dvs.nc", "dvs"),
    ],
    "S362ANI": [
        ("S362ANI_vs.nc", "vs"),
        ("S362ANI_vsh.nc", "vsh"),
        ("S362ANI_vsv.nc", "vsv"),
    ],
    "S362ANI+M": [
        ("S362ANI+M_vs.nc", "vs"),
        ("S362ANI+M_vsh.nc", "vsh"),
        ("S362ANI+M_vsv.nc", "vsv"),
    ],
    "S362WMANI": [
        ("S362WMANI_vs.nc", "vs"),
        ("S362WMANI_vsh.nc", "vsh"),
        ("S362WMANI_vsv.nc", "vsv"),
    ],
    "SAW24B16": [
        ("SAW24B16_vs.nc", "vs"),
    ],
    "SAW642AN": [
        ("SAW642AN_vs.nc", "vs"),
        ("SAW642AN_vp.nc", "vp"),
        ("SAW642AN_rho.nc", "rho"),
        ("SAW642AN_qs.nc", "qs"),
    ],
    "SAW642ANb": [
        ("SAW642ANb_vs.nc", "vs"),
        ("SAW642ANb_vp.nc", "vp"),
        ("SAW642ANb_rho.nc", "rho"),
        ("SAW642ANb_qs.nc", "qs"),
    ],
    "SEISGLOB2": [
        ("SEISGLOB2_dvs.nc", "dvs"),
    ],
    "SEMum": [
        ("SEMum_vs.nc", "vs"),
        ("SEMum_xi.nc", "xi"),
    ],
    "SEMUCB-WM1": [
        ("SEMUCB-WM1_dvs.nc", "dvs"),
        ("SEMUCB-WM1_vs.nc", "vs"),
        ("SEMUCB-WM1_vsh.nc", "vsh"),
        ("SEMUCB-WM1_vsv.nc", "vsv"),
    ],
    "SGLOBE-rani": [
        ("SGLOBE-rani_dvs.nc", "dvs"),
        ("SGLOBE-rani_vsh.nc", "vsh"),
        ("SGLOBE-rani_vsv.nc", "vsv"),
    ],
    "SP12RTS": [
        ("SP12RTS_dvp.nc", "dvp"),
        ("SP12RTS_dvs.nc", "dvs"),
    ],
    "SPani": [
        ("SPani_dvp.nc", "dvp"),
        ("SPani_dvs.nc", "dvs"),
        ("SPani_phi.nc", "phi"),
        ("SPani_vp.nc", "vp"),
        ("SPani_vs.nc", "vs"),
        ("SPani_xi.nc", "xi"),
    ],
    "TX2000": [
        ("TX2000_dvs.nc", "dvs"),
    ],
    "TX2011": [
        ("TX2011_dvs.nc", "dvs"),
        ("TX2011_vs.nc", "vs"),
    ],
    "TX2019slab": [
        ("TX2019slab_dvp.nc", "dvp"),
        ("TX2019slab_dvs.nc", "dvs"),
    ],
}


# ── GRD Model Registry ────────────────────────────────────────────────────
# Maps gdrift model name -> list of (directory_name_suffix, canonical_field_name).
# Directory names are relative to --grd-source-dir and have _abs appended.
# E.g. ("S_v", "vsv") means <grd-source-dir>/<ModelName>_S_v_abs/
GRD_MODEL_REGISTRY = {
    # ── Overlapping models (overwrite old data) ──
    "S40RTS": [
        ("S40RTS_S_v", "vsv"),
        ("S40RTS_S_h", "vsh"),
        ("S40RTS_S_i", "vs"),
    ],
    "S20RTS": [
        ("S20RTS_S_v", "vsv"),
        ("S20RTS_S_h", "vsh"),
        ("S20RTS_S_i", "vs"),
    ],
    "S362ANI": [
        ("S362ANI_S_v", "vsv"),
        ("S362ANI_S_h", "vsh"),
        ("S362ANI_S_i", "vs"),
    ],
    "S362WMANI": [
        ("S362WMANI_S_v", "vsv"),
        ("S362WMANI_S_h", "vsh"),
        ("S362WMANI_S_i", "vs"),
    ],
    "SEMUCB-WM1": [
        ("SEMUCB-WM1_S_v", "vsv"),
        ("SEMUCB-WM1_S_h", "vsh"),
        ("SEMUCB-WM1_S_i", "vs"),
    ],
    "SPani": [
        ("SPani_S_v", "vsv"),
        ("SPani_S_h", "vsh"),
        ("SPani_S_i", "vs"),
        ("SPani_P_v", "vpv"),
        ("SPani_P_h", "vph"),
        ("SPani_P_i", "vp"),
    ],
    "TX2011": [
        ("TX2011_S_i", "vs"),
    ],
    "GyPSuM": [
        ("GYPSUM_S_i", "vs"),
        ("GYPSUM_P_i", "vp"),
    ],
    # ── New models ──
    "3D2015-07Sv": [
        ("3D2015_07Sv_S_v", "vsv"),
    ],
    "AF2019": [
        ("AF2019_S_v", "vsv"),
    ],
    "ANT-20": [
        ("ANT-20_S_i", "vs"),
    ],
    "AuSREM": [
        ("AuSREM_S_v", "vsv"),
        ("AuSREM_S_h", "vsh"),
        ("AuSREM_P", "vp"),
    ],
    "Aus22": [
        ("Aus22_S_i", "vs"),
    ],
    "CAM2016": [
        ("CAM2016_S_v", "vsv"),
    ],
    "CSEM-2019": [
        ("CSEM_2019_S_v", "vsv"),
    ],
    "DETOX-P02": [
        ("DETOX-P02_P_i", "vp"),
    ],
    "F2010-Afr": [
        ("F2010_Afr_S_i", "vs"),
    ],
    "FR12": [
        ("FR12_S_i", "vs"),
        ("FR12_S_v", "vsv"),
    ],
    "LLNL-G3D-JPS": [
        ("LLNL-G3D-JPS_S_i", "vs"),
    ],
    "MITS-18": [
        ("MITS-18_S_i", "vs"),
    ],
    "PM13": [
        ("PM13_S_v", "vsv"),
    ],
    "PMEAN": [
        ("PMEAN_P_i", "vp"),
        ("PMEAN_P_h", "vph"),
        ("PMEAN_P_v", "vpv"),
    ],
    "SA2019": [
        ("SA2019_S_v", "vsv"),
    ],
    "SAVANI": [
        ("SAVANI_S_v", "vsv"),
        ("SAVANI_S_h", "vsh"),
        ("SAVANI_S_i", "vs"),
    ],
    "SEMUM2": [
        ("SEMUM2_S_v", "vsv"),
        ("SEMUM2_S_h", "vsh"),
        ("SEMUM2_S_i", "vs"),
    ],
    "SL2013NA": [
        ("SL2013NA_S_v", "vsv"),
    ],
    "SL2013sv": [
        ("SL2013sv_S_v", "vsv"),
    ],
    "SL2013sv-uninterp": [
        ("SL2013sv_uninterp_S_v", "vsv"),
    ],
    "SMEAN": [
        ("SMEAN_S_v", "vsv"),
        ("SMEAN_S_h", "vsh"),
        ("SMEAN_S_i", "vs"),
    ],
    "Y14": [
        ("Y14_S_v", "vsv"),
    ],
}

# Models that cover only a geographic sub-region
GRD_REGIONAL_MODELS = {
    "AF2019", "ANT-20", "AuSREM", "Aus22", "F2010-Afr",
    "FR12", "MITS-18", "SA2019", "Y14",
}


# ── NetCDF readers ──────────────────────────────────────────────────────────

def read_standard_netcdf(filepath):
    """Read a single-variable NetCDF file (depth, lat, lon, v).

    Returns
    -------
    data : dict
        Keys: 'depth', 'latitude', 'longitude', 'v' (3D array [depth, lat, lon])
    attrs : dict
        Global attributes from the NetCDF file.
    """
    dataset = nc.Dataset(str(filepath), "r", maskandscale=False)
    dataset.set_auto_maskandscale(False)

    data = {}
    for var in dataset.variables:
        data[var] = dataset.variables[var][:]

    # Normalise longitude to [-180, 180]
    data["longitude"][data["longitude"] > 180.0] -= 360.0

    attrs = {}
    for attr in dataset.ncattrs():
        attrs[attr] = getattr(dataset, attr)

    dataset.close()
    return data, attrs


def read_reveal_netcdf(filepath):
    """Read the multi-variable REVEAL.nc file.

    Returns
    -------
    data : dict
        Keys: 'depth', 'latitude', 'longitude', plus field names (vsv, vsh, vpv, rho).
    attrs : dict
        Global attributes.
    field_names : list of str
        The canonical field names found (e.g. ['vsv', 'vsh', 'vpv', 'rho']).
    """
    dataset = nc.Dataset(str(filepath), "r")

    data = {}
    field_names = []
    coord_vars = {"depth", "latitude", "longitude"}

    for var in dataset.variables:
        data[var] = dataset.variables[var][:]
        if var not in coord_vars:
            field_names.append(var)

    # Normalise longitude
    data["longitude"][data["longitude"] > 180.0] -= 360.0

    attrs = {}
    for attr in dataset.ncattrs():
        attrs[attr] = getattr(dataset, attr)

    dataset.close()
    return data, attrs, field_names


def _read_grd_coords(ds):
    """Extract lon/lat arrays from a GRD NetCDF dataset."""
    if "lon" in ds.variables:
        lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    elif "x" in ds.variables:
        lons = np.asarray(ds.variables["x"][:], dtype=np.float64)
        lats = np.asarray(ds.variables["y"][:], dtype=np.float64)
    else:
        raise KeyError(f"No lon/lat or x/y variables in dataset")
    # Normalise longitude 0-360 -> -180-180
    lons[lons > 180.0] -= 360.0
    return lons, lats


def read_grd_directory(dirpath):
    """Read a directory of per-depth GRD (NetCDF) files.

    Each file is named ``{depth}_km.grd`` or ``{depth}km.grd`` and contains
    a 2D lon/lat grid with a ``z`` data variable.  Variable names may be
    ``lon/lat`` or ``x/y`` (both geographic degrees).

    When depth files have different grid resolutions (e.g. shallow at 1-degree
    and deep at 0.5-degree), each slice is resampled to the reference grid
    (taken from the first valid file) via bilinear interpolation.

    Returns
    -------
    data : dict
        Keys: 'longitude', 'latitude', 'depth' (1D arrays) and
        'v' (3D array [depth, lat, lon]).  Masked/fill values are
        replaced with NaN.
    """
    from scipy.interpolate import RegularGridInterpolator

    dirpath = Path(dirpath)

    # Match both {depth}_km.grd and {depth}km.grd patterns
    grd_files = sorted(set(dirpath.glob("*_km.grd")) | set(dirpath.glob("*km.grd")))
    if not grd_files:
        raise FileNotFoundError(f"No *_km.grd or *km.grd files found in {dirpath}")

    # Parse depth from filename and filter
    depth_file_pairs = []
    for fp in grd_files:
        depth_str = fp.stem.replace("_km", "").replace("km", "")
        try:
            depth_km = float(depth_str)
        except ValueError:
            warnings.warn(f"Skipping unparseable filename: {fp.name}")
            continue
        if depth_km < 0:
            continue  # skip negative depths (ANT-20 artefacts)
        depth_file_pairs.append((depth_km, fp))

    depth_file_pairs.sort(key=lambda x: x[0])

    if not depth_file_pairs:
        raise ValueError(f"No valid depth files found in {dirpath}")

    # Read first file to get reference grid
    first_ds = nc.Dataset(str(depth_file_pairs[0][1]), "r")
    lons, lats = _read_grd_coords(first_ds)
    first_ds.close()

    nlat, nlon = len(lats), len(lons)
    n_resampled = 0

    depths = []
    slices = []

    for depth_km, fp in depth_file_pairs:
        ds = nc.Dataset(str(fp), "r")
        z = ds.variables["z"][:]

        # Check if this file has a different grid
        file_lons, file_lats = _read_grd_coords(ds)
        ds.close()

        # Handle masked arrays
        if isinstance(z, np.ma.MaskedArray):
            if z.mask.all():
                warnings.warn(f"Skipping fully-masked file: {fp.name}")
                continue
            z = np.where(z.mask, np.nan, z.data)
        z = np.asarray(z, dtype=np.float64)

        # Resample if grid dimensions differ from reference
        if z.shape != (nlat, nlon):
            interp = RegularGridInterpolator(
                (file_lats, file_lons), z,
                method="linear", bounds_error=False, fill_value=np.nan)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            z = interp((lat_grid, lon_grid))
            n_resampled += 1

        depths.append(depth_km)
        slices.append(z)

    if not depths:
        raise ValueError(f"All files masked/skipped in {dirpath}")

    if n_resampled > 0:
        warnings.warn(
            f"{n_resampled}/{len(depths)} depth files resampled to reference "
            f"grid ({nlat}x{nlon}) in {dirpath.name}")

    return {
        "longitude": lons,
        "latitude": lats,
        "depth": np.array(depths),
        "v": np.array(slices),  # shape (n_depths, nlat, nlon)
    }


# ── Core interpolation ──────────────────────────────────────────────────────

def interpolate_to_fibonacci(raw_data, fib_lons, fib_lats,
                             num_neighbours=10, far_threshold=5.0):
    """Interpolate lon/lat gridded data onto Fibonacci sphere points.

    Uses 1/d^2 (IDW power=2) weighting for smoother results than 1/d.

    Parameters
    ----------
    raw_data : dict
        Must contain 'longitude', 'latitude', 'depth', and 'v' (3D array).
    fib_lons, fib_lats : array
        Target Fibonacci point coordinates in degrees.
    num_neighbours : int
        Number of KD-tree neighbours.
    far_threshold : float
        Points with nearest-neighbour distance > this (degrees) get NaN.

    Returns
    -------
    result : ndarray, shape (num_layers * num_points,)
        Flattened interpolated values.
    num_layers : int
        Number of depth layers.
    """
    lons_grid, lats_grid = np.meshgrid(
        raw_data["longitude"], raw_data["latitude"], indexing="xy")

    tree = cKDTree(np.column_stack((lons_grid.ravel(), lats_grid.ravel())))
    dists, indices = tree.query(
        np.column_stack((fib_lons, fib_lats)), k=num_neighbours)

    num_layers = raw_data["depth"].shape[0]
    num_points = len(fib_lons)

    close_points = dists[:, 0] < 1e-10
    far_points = dists[:, 0] > far_threshold

    if np.any(far_points):
        warnings.warn(
            f"KD-tree found {np.sum(far_points)} points with nearest "
            f"distance > {far_threshold} degrees.", UserWarning)

    # Build value array: (num_layers, num_points, num_neighbours)
    values_at_neighbours = np.array(
        [raw_data["v"][layer, :, :].ravel()[indices]
         for layer in range(num_layers)])

    # 1/d^2 weights
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (dists ** 2)  # (num_points, num_neighbours)
        weight_sum = np.sum(weights, axis=1)  # (num_points,)

        # Weighted interpolation: einsum over neighbours
        # values_at_neighbours: (num_layers, num_points, num_neighbours)
        # weights: (num_points, num_neighbours)
        result = np.einsum(
            "lpk,pk->lp", values_at_neighbours, weights) / weight_sum

    # Exact values for coincident points
    result[:, close_points] = np.array(
        [raw_data["v"][layer, :, :].ravel()[indices[close_points, 0]]
         for layer in range(num_layers)])

    # NaN for far points
    result[:, far_points] = np.nan

    return result.ravel(), num_layers


def interpolate_reveal_field(field_data, tree, dists, indices,
                             close_points, far_points, num_layers):
    """Interpolate a single 3D field from REVEAL onto Fibonacci points.

    Same logic as interpolate_to_fibonacci but reuses the pre-built tree/indices.
    """
    values_at_neighbours = np.array(
        [field_data[layer, :, :].ravel()[indices]
         for layer in range(num_layers)])

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (dists ** 2)
        weight_sum = np.sum(weights, axis=1)
        result = np.einsum(
            "lpk,pk->lp", values_at_neighbours, weights) / weight_sum

    result[:, close_points] = np.array(
        [field_data[layer, :, :].ravel()[indices[close_points, 0]]
         for layer in range(num_layers)])

    result[:, far_points] = np.nan

    return result.ravel()


# ── Coordinate construction ────────────────────────────────────────────────

def build_coordinates(fib_coords, depths_km):
    """Build Cartesian coordinates for all depth layers.

    Parameters
    ----------
    fib_coords : ndarray, shape (num_points, 3)
        Unit-sphere Fibonacci points.
    depths_km : ndarray, shape (num_layers,)
        Depth values in km.

    Returns
    -------
    coordinates : ndarray, shape (num_layers * num_points, 3)
    """
    coords_list = [
        fib_coords * (R_earth - depth * 1e3)
        for depth in depths_km
    ]
    return np.vstack(coords_list)


# ── HDF5 writer ─────────────────────────────────────────────────────────────

def write_hdf5(output_path, data_dict, metadata):
    """Write flat HDF5 with coordinates and field arrays.

    Parameters
    ----------
    output_path : Path
        Full path to the output .h5 file.
    data_dict : dict
        'coordinates': (N,3) array, plus field_name: (N,) arrays.
    metadata : dict
        File-level attributes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)
        for key, value in metadata.items():
            f.attrs[key] = value


def file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


# ── Single-model conversion ────────────────────────────────────────────────

def convert_single_model(model_name, source_dir, output_dir,
                         num_samples=65341, num_neighbours=10):
    """Convert one seismic tomography model from NetCDF to HDF5.

    Returns
    -------
    output_path : Path
        Path to the written HDF5 file.
    fields : list of str
        Canonical field names in the output.
    """
    config = MODEL_REGISTRY[model_name]

    # Generate Fibonacci sphere
    fib_coords = fibonacci_sphere(num_samples)
    fib_lats, fib_lons, _ = cartesian_to_geodetic(*fib_coords.T)

    data_to_write = {}
    all_metadata = {}
    coordinates = None
    fields_written = []

    if config == "REVEAL":
        # Special case: multi-variable file
        filepath = source_dir / "REVEAL.nc"
        if not filepath.exists():
            raise FileNotFoundError(f"REVEAL.nc not found in {source_dir}")

        data, attrs, field_names = read_reveal_netcdf(filepath)
        all_metadata = attrs

        num_layers = data["depth"].shape[0]
        coordinates = build_coordinates(fib_coords, data["depth"])

        # Build KD-tree once
        lons_grid, lats_grid = np.meshgrid(
            data["longitude"], data["latitude"], indexing="xy")
        tree = cKDTree(np.column_stack((lons_grid.ravel(), lats_grid.ravel())))
        dists, indices = tree.query(
            np.column_stack((fib_lons, fib_lats)), k=num_neighbours)

        close_points = dists[:, 0] < 1e-10
        far_points = dists[:, 0] > 5.0

        for field in field_names:
            interpolated = interpolate_reveal_field(
                data[field], tree, dists, indices,
                close_points, far_points, num_layers)

            # Scale absolute velocities to m/s
            if field in SCALE_TO_MS:
                interpolated = interpolated * 1e3

            data_to_write[field] = interpolated
            fields_written.append(field)

    else:
        # Standard case: one or more single-variable files
        for filename, field_name in config:
            filepath = source_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"{filename} not found in {source_dir}")

            raw_data, attrs = read_standard_netcdf(filepath)
            if not all_metadata:
                all_metadata = attrs

            interpolated, num_layers = interpolate_to_fibonacci(
                raw_data, fib_lons, fib_lats, num_neighbours=num_neighbours)

            # Build coordinates from the first file
            if coordinates is None:
                coordinates = build_coordinates(fib_coords, raw_data["depth"])
            else:
                # Verify coordinate consistency
                expected = build_coordinates(fib_coords, raw_data["depth"])
                if not np.allclose(coordinates, expected):
                    warnings.warn(
                        f"{model_name}: coordinate mismatch between files. "
                        f"Using coordinates from the first file.",
                        UserWarning)

            # Scale absolute velocities to m/s
            if field_name in SCALE_TO_MS:
                interpolated = interpolated * 1e3

            data_to_write[field_name] = interpolated
            fields_written.append(field_name)

    # Add coordinates
    data_to_write["coordinates"] = coordinates

    # Update metadata
    all_metadata["comment"] = all_metadata.get("comment", "") + (
        "\nResampled to Fibonacci sphere ({n} points per layer, "
        "{k} neighbours, IDW power=2) by Sia Ghelichkhan "
        "(siavash.ghelichkhan@anu.edu.au)".format(
            n=num_samples, k=num_neighbours)
    )
    all_metadata["velocity_units"] = "m/s"
    all_metadata["perturbation_units"] = "fractional (model-dependent)"

    # Write output
    dataset_name = f"3d_seismic_{model_name}"
    h5_filename = f"{hash_name(dataset_name)}.h5"
    output_path = output_dir / h5_filename

    write_hdf5(output_path, data_to_write, all_metadata)

    # Sanity checks
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  Output: {output_path}")
    print(f"  Fields: {sorted(fields_written)}")
    print(f"  Coordinate shape: {coordinates.shape}")
    for field in sorted(fields_written):
        arr = data_to_write[field]
        n_nan = np.sum(np.isnan(arr))
        print(f"  {field:>6s}: min={np.nanmin(arr):12.4f}  "
              f"max={np.nanmax(arr):12.4f}  "
              f"mean={np.nanmean(arr):12.4f}  "
              f"NaN={n_nan}")
        if n_nan == len(arr):
            warnings.warn(f"{model_name}/{field} is ALL NaN!", UserWarning)

    return output_path, sorted(fields_written)


def convert_grd_model(model_name, source_dir, output_dir,
                      num_samples=65341, num_neighbours=10):
    """Convert one GRD-collection model (multi-directory) to HDF5.

    Handles components with inconsistent depth levels by computing their
    union and NaN-filling missing depths per component.

    Returns
    -------
    output_path : Path
    fields : list of str
    """
    config = GRD_MODEL_REGISTRY[model_name]

    # ── Phase 1: read all component directories ──
    component_data = {}  # field_name -> dict with depth/longitude/latitude/v
    all_depth_sets = {}  # field_name -> set of depths

    for dir_prefix, field_name in config:
        dirpath = source_dir / f"{dir_prefix}_abs"
        if not dirpath.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {dirpath}")
        data = read_grd_directory(dirpath)
        component_data[field_name] = data
        all_depth_sets[field_name] = set(data["depth"].tolist())
        print(f"  {field_name} <- {dir_prefix}_abs: "
              f"{len(data['depth'])} depths, "
              f"grid {len(data['latitude'])}x{len(data['longitude'])}")

    # ── Phase 2: compute union of all depth levels ──
    union_depths = sorted(set().union(*all_depth_sets.values()))
    union_depths = np.array(union_depths)
    print(f"  Union depths: {len(union_depths)} levels "
          f"({union_depths[0]:.0f} - {union_depths[-1]:.0f} km)")

    # ── Phase 3: generate Fibonacci sphere once ──
    fib_coords = fibonacci_sphere(num_samples)
    fib_lats, fib_lons, _ = cartesian_to_geodetic(*fib_coords.T)
    num_points = len(fib_lons)
    num_layers = len(union_depths)

    coordinates = build_coordinates(fib_coords, union_depths)

    # ── Phase 4: interpolate each component ──
    data_to_write = {"coordinates": coordinates}
    fields_written = []

    for field_name, data in component_data.items():
        component_depths = set(data["depth"].tolist())

        # Build KD-tree for this component's grid
        lons_grid, lats_grid = np.meshgrid(
            data["longitude"], data["latitude"], indexing="xy")
        tree = cKDTree(np.column_stack((lons_grid.ravel(), lats_grid.ravel())))
        dists, indices = tree.query(
            np.column_stack((fib_lons, fib_lats)), k=num_neighbours)

        close_points = dists[:, 0] < 1e-10
        far_points = dists[:, 0] > 5.0

        # Pre-compute weights
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (dists ** 2)
            weight_sum = np.sum(weights, axis=1)

        # Map component depth index -> union depth index
        comp_depth_to_idx = {d: i for i, d in enumerate(data["depth"].tolist())}

        result = np.full((num_layers, num_points), np.nan)

        for ui, udepth in enumerate(union_depths.tolist()):
            if udepth not in component_depths:
                continue  # leave as NaN

            ci = comp_depth_to_idx[udepth]
            layer_data = data["v"][ci, :, :]

            # Check for NaN in source (partially masked regional data)
            flat = layer_data.ravel()
            vals_at_nb = flat[indices]  # (num_points, num_neighbours)

            # IDW interpolation
            with np.errstate(divide='ignore', invalid='ignore'):
                # NaN-aware: mask NaN neighbours
                nan_mask = np.isnan(vals_at_nb)
                masked_vals = np.where(nan_mask, 0.0, vals_at_nb)
                masked_weights = np.where(nan_mask, 0.0, weights)
                masked_weight_sum = np.sum(masked_weights, axis=1)

                row = np.where(
                    masked_weight_sum > 0,
                    np.sum(masked_vals * masked_weights, axis=1) / masked_weight_sum,
                    np.nan,
                )

            # Exact values for coincident points
            row[close_points] = flat[indices[close_points, 0]]

            # NaN for far-from-data points
            row[far_points] = np.nan

            result[ui, :] = row

        # Scale km/s -> m/s for absolute velocity fields
        if field_name in SCALE_TO_MS:
            result = result * 1e3

        data_to_write[field_name] = result.ravel()
        fields_written.append(field_name)

    # ── Phase 5: metadata and write ──
    metadata = {
        "comment": (
            "Converted from GRD per-depth files. "
            "Resampled to Fibonacci sphere ({n} points per layer, "
            "{k} neighbours, IDW power=2) by Sia Ghelichkhan "
            "(siavash.ghelichkhan@anu.edu.au)".format(
                n=num_samples, k=num_neighbours)
        ),
        "velocity_units": "m/s",
        "num_depth_levels": int(num_layers),
        "num_points_per_level": int(num_points),
    }
    if model_name in GRD_REGIONAL_MODELS:
        metadata["regional"] = "true"

    dataset_name = f"3d_seismic_{model_name}"
    h5_filename = f"{hash_name(dataset_name)}.h5"
    output_path = output_dir / h5_filename

    write_hdf5(output_path, data_to_write, metadata)

    # ── Sanity checks ──
    print(f"\n{'='*60}")
    print(f"Model: {model_name} (GRD)")
    print(f"  Output: {output_path}")
    print(f"  Fields: {sorted(fields_written)}")
    print(f"  Coordinate shape: {coordinates.shape}")
    for field in sorted(fields_written):
        arr = data_to_write[field]
        n_nan = np.sum(np.isnan(arr))
        n_total = len(arr)
        print(f"  {field:>6s}: min={np.nanmin(arr):12.4f}  "
              f"max={np.nanmax(arr):12.4f}  "
              f"mean={np.nanmean(arr):12.4f}  "
              f"NaN={n_nan} ({100*n_nan/n_total:.1f}%)")
        if n_nan == n_total:
            warnings.warn(f"{model_name}/{field} is ALL NaN!", UserWarning)

    return output_path, sorted(fields_written)


# ── Manifest generation ────────────────────────────────────────────────────

def load_existing_manifest(manifest_path):
    """Load existing datasets.json to get citation info."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def generate_manifest_entries(results, manifest_path, regional_models=None):
    """Generate JSON entries with updated hashes and fields.

    Parameters
    ----------
    results : list of (model_name, output_path, fields)
    manifest_path : Path
        Path to the existing datasets.json.
    regional_models : set, optional
        Model names that are regional (not global coverage).

    Returns
    -------
    entries : list of dict
    """
    if regional_models is None:
        regional_models = set()

    manifest = load_existing_manifest(manifest_path)
    existing = {e["name"]: e for e in manifest["datasets"]}

    entries = []
    for model_name, output_path, fields in results:
        dataset_name = f"3d_seismic_{model_name}"
        sha = file_hash(output_path)

        if dataset_name in existing:
            entry = dict(existing[dataset_name])
            entry["sha256"] = sha
            entry["fields"] = fields
        else:
            entry = {
                "name": dataset_name,
                "filename": output_path.name,
                "type": "TOMOGRAPHY_MODEL",
                "utility": "SEISMIC_MODEL",
                "source": "UNKNOWN - please add citation",
                "sha256": sha,
                "fields": fields,
            }

        if model_name in regional_models:
            entry["regional"] = True

        entries.append(entry)

    return entries


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert seismic tomography NetCDF files to HDF5 point clouds.")
    parser.add_argument(
        "--source-dir", type=Path, default=None,
        help="Directory containing the raw Mather-collection NetCDF files.")
    parser.add_argument(
        "--grd-source-dir", type=Path, default=None,
        help="Directory containing GRD per-depth tomography directories.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gdrift/data"),
        help="Directory for output HDF5 files (default: gdrift/data).")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Specific Mather-collection model names to convert.")
    parser.add_argument(
        "--grd-models", nargs="*", default=None,
        help="Specific GRD-collection model names to convert.")
    parser.add_argument(
        "--num-samples", type=int, default=181 * 361,
        help="Number of Fibonacci sphere points (default: 65341).")
    parser.add_argument(
        "--num-neighbours", type=int, default=10,
        help="Number of KD-tree neighbours for IDW (default: 10).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.source_dir is None and args.grd_source_dir is None:
        raise ValueError("At least one of --source-dir or --grd-source-dir is required")

    results = []
    failed = []
    manifest_path = Path(__file__).resolve().parent.parent / "gdrift" / "datasets.json"

    # ── Mather collection ──
    if args.source_dir is not None:
        mather_models = args.models or list(MODEL_REGISTRY.keys())
        for m in mather_models:
            if m not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown Mather model '{m}'. "
                    f"Available: {list(MODEL_REGISTRY.keys())}")

        print(f"Converting {len(mather_models)} Mather-collection models")
        print(f"Source: {args.source_dir}")
        print(f"Output: {args.output_dir}")
        print(f"Fibonacci points: {args.num_samples}")
        print(f"Neighbours: {args.num_neighbours}")

        for model_name in mather_models:
            try:
                output_path, fields = convert_single_model(
                    model_name,
                    source_dir=args.source_dir,
                    output_dir=args.output_dir,
                    num_samples=args.num_samples,
                    num_neighbours=args.num_neighbours,
                )
                results.append((model_name, output_path, fields))
            except Exception as e:
                print(f"\nFAILED: {model_name}: {e}")
                failed.append((model_name, str(e)))

    # ── GRD collection ──
    if args.grd_source_dir is not None:
        grd_models = args.grd_models or list(GRD_MODEL_REGISTRY.keys())
        for m in grd_models:
            if m not in GRD_MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown GRD model '{m}'. "
                    f"Available: {list(GRD_MODEL_REGISTRY.keys())}")

        print(f"\nConverting {len(grd_models)} GRD-collection models")
        print(f"Source: {args.grd_source_dir}")
        print(f"Output: {args.output_dir}")

        for model_name in grd_models:
            try:
                output_path, fields = convert_grd_model(
                    model_name,
                    source_dir=args.grd_source_dir,
                    output_dir=args.output_dir,
                    num_samples=args.num_samples,
                    num_neighbours=args.num_neighbours,
                )
                results.append((model_name, output_path, fields))
            except Exception as e:
                print(f"\nFAILED: {model_name}: {e}")
                failed.append((model_name, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed models:")
        for name, err in failed:
            print(f"  - {name}: {err}")

    # Generate manifest entries
    if results:
        entries = generate_manifest_entries(
            results, manifest_path, regional_models=GRD_REGIONAL_MODELS)

        out_manifest = Path("new_seismic_manifest_entries.json")
        with open(out_manifest, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"\nManifest entries saved to: {out_manifest}")
        print("Manually merge these into gdrift/datasets.json")


if __name__ == "__main__":
    main()
