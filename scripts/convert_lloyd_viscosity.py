#!/usr/bin/env python3
"""Convert the 3-D mantle viscosity model of Lloyd et al. (2024) to a gdrift dataset.

The model is the "3-D viscosity inference" of Lloyd et al. (2024), GJI 236,
1139-1171, doi:10.1093/gji/ggad455. It is not an inversion of GIA data. The
authors convert the shear-wave speeds of GLAD-M25 to temperature and then to
steady-state diffusion-creep viscosity: above 400 km with the calibration of
Richards et al. (2020), below 400 km with the approach of Austermann et al.
(2021). The model is scaled to mean viscosities of 5e20 Pa s in the
sublithospheric upper mantle and 5e21 Pa s in the lower mantle.

Source file
-----------
The paper's supporting information contains `3D_visc_infer_GLAD-M25.nc`
(NetCDF3, written by A. J. Lloyd in November 2023). The article is open
access under CC BY 4.0. Europe PMC serves the supporting information of the
PMC copy (PMC10753356) as a zip file that contains a second zip file, which
contains a tar.gz with the NetCDF file. The zip files are built on request,
so their sha256 changes. The script therefore pins the sha256 of the NetCDF
file itself (SOURCE_SHA256) and refuses any other file.

The NetCDF file has one variable, `log10eta(depth, latitude, longitude)`, in
log10(Pa s), on a regular grid: 201 depths from 0 to 2891 km (spacing 1, 5,
25, 41 and 50 km), latitudes -90 to 90 and longitudes -180 to 180, both every
0.5 degrees.

Output
------
One flat HDF5 file in the layout of the gdrift seismic models:

- `coordinates`: (N, 3) Cartesian positions in metres. The points are the
  same 65,341-point Fibonacci lattice as the seismic models, repeated on each
  of the 201 depths of the source file, surface layer first.
- `log10_viscosity`: (N,) log10 of the viscosity in Pa s.

The field is stored as log10 because the values range over 21 orders of
magnitude (1e18 to 1e39 Pa s). Any interpolation of the linear values
between grid points would be dominated by the largest neighbour. gdrift
interpolates the stored field directly, so the stored field must be the one
that is safe to interpolate.

The values above 1e30 Pa s in the lithosphere (12 to about 150 km depth) are
kept as published. GIA treats these cells as elastic. A user who needs a cap
can clip the interpolated log10 values.

Resampling
----------
The source grid is regular in latitude and longitude, so the script uses
bilinear interpolation in (latitude, longitude) on each depth layer, applied
to the log10 values. This is exact at grid nodes, has no problem at the
date line (the source includes both -180 and +180), and does not need the
neighbour search in degree space that scripts/convert_seismic_models.py uses
for irregular sources. No interpolation is done in depth: the output keeps
the 201 source depths.

Usage
-----
    python scripts/convert_lloyd_viscosity.py                    # download, convert, check, report
    python scripts/convert_lloyd_viscosity.py --source FILE.nc   # use a local copy of the NetCDF file
    python scripts/convert_lloyd_viscosity.py --write            # also write the files and the manifest

With --write, the file is saved twice: as gdrift/data-sia/<dataset name>.h5
(the input of scripts/upload_to_s3.py) and as gdrift/data/<sha256>.h5 (where
the loader finds it before the upload). The manifest gets a new entry, or an
updated one, with the sha256, the file name and `etag: null` until the upload.
See docs/dataset-releases.md.
"""
import argparse
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy.io import netcdf_file

# Reuse the package functions, so that the coordinates and the sha256 are
# computed exactly as in the other datasets and in the loader.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gdrift.constants import R_earth  # noqa: E402
from gdrift.io import file_hash  # noqa: E402
from gdrift.utility import cartesian_to_geodetic, fibonacci_sphere  # noqa: E402

MANIFEST_PATH = REPO / "gdrift" / "datasets.json"
DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

# Name of the dataset in the manifest. The prefix "3d_earthmodel_" marks a
# 3-D model that is not seismic tomography; such models load through
# gdrift.EarthModel3D.from_dataset(<name without prefix>).
DATASET_NAME = "3d_earthmodel_Lloyd_et_al_2024_GJI"

# Name of the field in the output file and the unit that goes with it.
FIELD_NAME = "log10_viscosity"
FIELD_UNITS = "log10(Pa s)"

# Supporting information of the PMC copy of the paper, served by Europe PMC.
SOURCE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10753356/supplementaryFiles"
INNER_ZIP = "ggad455_supplemental_files.zip"
TARBALL = "3D_visc_infer_GLAD-M25.nc.tar.gz"
SOURCE_FILE = "3D_visc_infer_GLAD-M25.nc"
# sha256 of the NetCDF file as published (checked on 2026-10-18).
SOURCE_SHA256 = "d29d17a775ba8cded080e2d8af67e029ec6f7f786f3c2a7ada7c4d5159c2a69d"

# Same lateral sampling as the seismic models (scripts/convert_seismic_models.py).
NUM_POINTS_PER_LAYER = 65341

CITATION = (
    "Lloyd, A. J., Crawford, O., Al-Attar, D., Austermann, J., Hoggard, M. J., "
    "Richards, F. D., & Syvret, F. (2024). GIA imaging of 3-D mantle viscosity "
    "based on palaeo sea level observations - Part I: Sensitivity kernels for an "
    "Earth with laterally varying viscosity. Geophysical Journal International, "
    "236(2), 1139-1171."
)
DOI = "10.1093/gji/ggad455"

DESCRIPTION = (
    "Global 3D mantle viscosity (log10 Pa s) inferred from the shear-wave speeds of "
    "GLAD-M25, not inverted from GIA data. Above 400 km it uses the calibration of "
    "Richards et al. (2020), below 400 km the approach of Austermann et al. (2021). "
    "Mean viscosity is 5e20 Pa s in the sublithospheric upper mantle and 5e21 Pa s "
    "in the lower mantle. Lithosphere values above 1e30 Pa s are kept as published."
)


def download_source(directory):
    """Download the supporting information and extract the NetCDF file.

    Parameters
    ----------
    directory : Path
        Empty directory for the extracted file.

    Returns
    -------
    Path
        Path of the extracted NetCDF file.
    """
    print(f"Downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=600) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    # The outer zip holds the figures and a second zip with the supplement
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    # The second zip holds the supplementary figures and the tarball
    with tarfile.open(fileobj=io.BytesIO(inner.read(TARBALL)), mode="r:gz") as tar:
        member = tar.getmember(SOURCE_FILE)
        # Read the one regular file by name; never extract arbitrary paths
        if not member.isfile():
            raise RuntimeError(f"{SOURCE_FILE} in {TARBALL} is not a regular file")
        path = directory / SOURCE_FILE
        with tar.extractfile(member) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return path


def read_source(path):
    """Read the NetCDF file and check its grid.

    Parameters
    ----------
    path : Path
        The NetCDF file. Its sha256 must equal SOURCE_SHA256.

    Returns
    -------
    depth_km : ndarray, shape (n_depth,)
        Depths below the surface in km, increasing.
    lat : ndarray, shape (n_lat,)
        Latitudes in degrees, increasing from -90 to 90.
    lon : ndarray, shape (n_lon,)
        Longitudes in degrees, increasing from -180 to 180.
    log10eta : ndarray, shape (n_depth, n_lat, n_lon)
        log10 of the viscosity in Pa s.
    """
    # file_hash returns "sha256:<hex>", the format of the manifest checks
    sha = file_hash(path).split(":", 1)[1]
    if sha != SOURCE_SHA256:
        raise RuntimeError(f"{path} has sha256 {sha}, expected {SOURCE_SHA256}")

    # The file is NetCDF3 classic, which scipy reads without netCDF4.
    # mmap=False copies the arrays, so they stay valid after the file closes.
    with netcdf_file(path, "r", mmap=False) as nc:
        depth_km = np.array(nc.variables["depth"][:], dtype=np.float64)
        lat = np.array(nc.variables["latitude"][:], dtype=np.float64)
        lon = np.array(nc.variables["longitude"][:], dtype=np.float64)
        log10eta = np.array(nc.variables["log10eta"][:], dtype=np.float64)

    # The bilinear interpolation below assumes a regular, increasing grid
    # that covers the whole sphere, with both -180 and +180 present.
    assert log10eta.shape == (depth_km.size, lat.size, lon.size), log10eta.shape
    assert np.all(np.diff(depth_km) > 0), "depths must increase"
    assert depth_km[0] == 0.0, "the first layer must be the surface"
    assert np.allclose(np.diff(lat), lat[1] - lat[0]), "latitudes must be evenly spaced"
    assert np.allclose(np.diff(lon), lon[1] - lon[0]), "longitudes must be evenly spaced"
    assert lat[0] == -90.0 and lat[-1] == 90.0, (lat[0], lat[-1])
    assert lon[0] == -180.0 and lon[-1] == 180.0, (lon[0], lon[-1])
    assert np.all(np.isfinite(log10eta)), "the source contains NaN or inf"
    # -180 and +180 are the same meridian. In the published file the two
    # columns are identical below about 180 km, but differ in the crust and
    # lithosphere (up to 6 log10 units at 21 km, mean 0.003), where the
    # authors assign crustal and lithospheric viscosities per grid column.
    # The file is converted as published: the bilinear interpolation uses the
    # -180 column for longitudes just east of -180 and the +180 column for
    # longitudes just west of +180, so each side keeps its own values. The
    # check below makes sure that the difference stays confined to the top
    # 200 km, which is what the published file shows.
    seam = np.abs(log10eta[:, :, 0] - log10eta[:, :, -1])
    assert seam[depth_km >= 200.0].max() == 0.0, "columns at -180 and +180 differ below 200 km"
    print(f"Columns at -180 and +180 differ above 200 km: max {seam.max():.2f}, "
          f"mean {seam.mean():.4f} log10 units")
    return depth_km, lat, lon, log10eta


def bilinear_weights(lat, lon, query_lat, query_lon):
    """Bilinear interpolation indices and weights on a regular lat-lon grid.

    The same indices and weights apply to every depth layer, so they are
    computed once.

    Parameters
    ----------
    lat, lon : ndarray
        Increasing, evenly spaced grid axes in degrees.
    query_lat, query_lon : ndarray, shape (n,)
        Query points in degrees, inside the grid.

    Returns
    -------
    i0, j0 : ndarray of int, shape (n,)
        Index of the grid cell (lower-left corner) of each query point.
    weights : ndarray, shape (n, 4)
        Weights of the corners (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1).
        Each row sums to 1.
    """
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    # Fractional grid index of each query point
    fi = (query_lat - lat[0]) / dlat
    fj = (query_lon - lon[0]) / dlon
    # Lower corner. Clip so that a point on the last grid line uses the last
    # cell with a fractional offset of 1 instead of reading past the end.
    i0 = np.clip(np.floor(fi).astype(int), 0, lat.size - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, lon.size - 2)
    ti = fi - i0
    tj = fj - j0
    weights = np.column_stack([
        (1 - ti) * (1 - tj),
        ti * (1 - tj),
        (1 - ti) * tj,
        ti * tj,
    ])
    return i0, j0, weights


def resample(depth_km, lat, lon, log10eta):
    """Resample the source grid onto the Fibonacci lattice of each depth.

    Parameters
    ----------
    depth_km, lat, lon, log10eta : ndarray
        As returned by `read_source`.

    Returns
    -------
    coordinates : ndarray, shape (n_depth * NUM_POINTS_PER_LAYER, 3)
        Cartesian positions in metres, surface layer first.
    values : ndarray, shape (n_depth * NUM_POINTS_PER_LAYER,)
        log10 viscosity at each position.
    """
    # Unit-sphere Fibonacci lattice, identical to the seismic models
    unit = fibonacci_sphere(NUM_POINTS_PER_LAYER)
    # Latitude and longitude of each lattice point. The depth output is
    # meaningless for unit vectors and is discarded.
    fib_lat, fib_lon, _ = cartesian_to_geodetic(unit[:, 0], unit[:, 1], unit[:, 2])

    i0, j0, w = bilinear_weights(lat, lon, fib_lat, fib_lon)
    # Corner values for all layers at once: shape (n_depth, n_points, 4)
    corners = np.stack([
        log10eta[:, i0, j0],
        log10eta[:, i0 + 1, j0],
        log10eta[:, i0, j0 + 1],
        log10eta[:, i0 + 1, j0 + 1],
    ], axis=-1)
    values = np.einsum("dpk,pk->dp", corners, w)

    # Radius of each layer in metres; R_earth is the gdrift reference radius
    radii = R_earth - depth_km * 1e3
    coordinates = np.vstack([unit * r for r in radii])
    return coordinates, values.ravel()


def build_file(path, coordinates, values, n_depth):
    """Write the HDF5 file in the layout of the seismic models.

    Parameters
    ----------
    path : Path
        Output file.
    coordinates : ndarray, shape (N, 3)
        Cartesian positions in metres.
    values : ndarray, shape (N,)
        log10 viscosity in Pa s.
    n_depth : int
        Number of depth layers.
    """
    attrs = {
        "comment": (
            f"Converted from {SOURCE_FILE} (supporting information of Lloyd et al., 2024). "
            f"Resampled to Fibonacci sphere ({NUM_POINTS_PER_LAYER} points per layer) by "
            "bilinear interpolation in latitude and longitude of log10 viscosity, on the "
            "201 source depths, by Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)"
        ),
        "num_depth_levels": np.int64(n_depth),
        "num_points_per_level": np.int64(NUM_POINTS_PER_LAYER),
        "viscosity_units": FIELD_UNITS,
        "model_reference": CITATION,
        "doi": DOI,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "licence": "CC BY 4.0 (open-access article and supporting information)",
        "note": (
            "Not an inversion of GIA data: viscosity inferred from GLAD-M25 shear-wave "
            "speeds. Values above 1e30 Pa s in the lithosphere are kept as published."
        ),
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("coordinates", data=coordinates)
        f.create_dataset(FIELD_NAME, data=values)
        for key, value in attrs.items():
            f.attrs[key] = value


def check_against_source(coordinates, values, depth_km, lat, lon, log10eta):
    """Compare the resampled field with the source and print a summary.

    Checks that the values stay inside the range of each source layer
    (bilinear interpolation cannot overshoot) and that the mean of each
    layer agrees with an area-weighted mean of the source layer.
    """
    n = NUM_POINTS_PER_LAYER
    n_depth = depth_km.size
    layers = values.reshape(n_depth, n)
    # Every radius in a layer must equal the radius of its source depth
    radii = np.linalg.norm(coordinates, axis=1).reshape(n_depth, n)
    expected = (R_earth - depth_km * 1e3)[:, None]
    assert np.allclose(radii, expected, rtol=0, atol=1e-3), "layer radii do not match the source depths"

    # Area weights of the source grid (cos latitude), for a fair mean
    wlat = np.cos(np.radians(lat))[:, None] * np.ones(lon.size - 1)[None, :]
    worst = 0.0
    print(f"{'depth km':>9} {'min':>7} {'mean':>7} {'src mean':>8} {'max':>7}")
    for k in range(n_depth):
        src = log10eta[k, :, :-1]  # drop the repeated +180 column
        assert layers[k].min() >= src.min() - 1e-9 and layers[k].max() <= src.max() + 1e-9, depth_km[k]
        src_mean = np.sum(src * wlat) / np.sum(wlat)
        worst = max(worst, abs(layers[k].mean() - src_mean))
        if depth_km[k] in (0, 30, 100, 200, 400, 670, 1000, 2000, 2891) or k == n_depth - 1:
            print(f"{depth_km[k]:9.0f} {layers[k].min():7.2f} {layers[k].mean():7.2f} "
                  f"{src_mean:8.2f} {layers[k].max():7.2f}")
    print(f"Largest difference of layer means (log10 units): {worst:.4f}")


def manifest_entry(sha):
    """Return the manifest entry for the new file."""
    return {
        "name": DATASET_NAME,
        "filename": f"{sha}.h5",
        "type": "EARTH_MODEL_3D",
        "utility": "EARTH_MODEL_3D",
        "source": CITATION,
        "doi": DOI,
        "year": 2024,
        "fields": [FIELD_NAME],
        "description": DESCRIPTION,
        "author": "Lloyd, A. J., Crawford, O., Al-Attar, D., Austermann, J., Hoggard, M. J., Richards, F. D., & Syvret, F.",
        "sha256": sha,
        "metadata_source": "web_search",
        "metadata_complete": True,
        "etag": None,
    }


def update_manifest(entry):
    """Add the entry to datasets.json, or replace the entry with the same name."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    names = [d["name"] for d in manifest["datasets"]]
    if entry["name"] in names:
        manifest["datasets"][names.index(entry["name"])] = entry
    else:
        manifest["datasets"].append(entry)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Local copy of 3D_visc_infer_GLAD-M25.nc")
    parser.add_argument("--write", action="store_true", help="Write the files and update datasets.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = args.source if args.source else download_source(tmp)
        depth_km, lat, lon, log10eta = read_source(source)
        print(f"Source: {depth_km.size} depths x {lat.size} lat x {lon.size} lon")

        coordinates, values = resample(depth_km, lat, lon, log10eta)
        check_against_source(coordinates, values, depth_km, lat, lon, log10eta)

        out = tmp / "out.h5"
        build_file(out, coordinates, values, depth_km.size)
        sha = file_hash(out).split(":", 1)[1]
        print(f"Output: {coordinates.shape[0]} points, {out.stat().st_size / 1e6:.0f} MB, sha256 {sha}")

        if not args.write:
            print("Dry run: nothing written. Use --write to save the files and update the manifest.")
            return

        DATA_SIA_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, DATA_SIA_DIR / f"{DATASET_NAME}.h5")
        shutil.copyfile(out, DATA_DIR / f"{sha}.h5")
        update_manifest(manifest_entry(sha))
        print(f"Wrote gdrift/data-sia/{DATASET_NAME}.h5, gdrift/data/{sha}.h5 and the manifest entry")


if __name__ == "__main__":
    main()
