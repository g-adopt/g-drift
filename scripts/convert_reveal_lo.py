#!/usr/bin/env python3
"""Convert the RevealLO seismic model to a gdrift dataset.

RevealLO (Schiller, Keating, Afanasiev, Marty and Fichtner, 2026,
doi:10.5281/zenodo.23646069) updates the REVEAL global full-waveform
inversion model (Thrastarson et al., 2024, the gdrift dataset
3d_seismic_REVEAL) with long-period minor- and major-arc surface waves and
multi-orbit body waves, at a shortest period of 33 s. The dataset has the
fields of 3d_seismic_REVEAL: vsv, vsh, vpv and the density rho.

This script uses the resampling, the checks and the file writer of
scripts/convert_glad_m35.py, as scripts/convert_glad_m25.py does, so that
the dataset is built in the same way as the two GLAD datasets. Only the
reading of the source, the selection of the depths and the metadata are
specific to RevealLO.

Source file
-----------
The Zenodo record has the model in two forms: the simulation mesh of the
spectral-element solver (Reveal_33s_with_LO.h5, 23 GB) and the model
sampled on a regular grid (Reveal_33s_with_LO_nan_filled.nc, 5.3 GB,
NetCDF4, which h5py reads as HDF5). The script reads the regular grid. It
pins the sha256 of the file (SOURCE_SHA256) and refuses any other file. The
file has:

- the fields VSV, VSH, VPV in m/s and RHO in kg/m^3, float64, with the
  dimensions (latitude, longitude, depth),
- 361 latitudes from -90 to 90 and 721 longitudes from 0 to 360, every
  0.5 degree,
- 638 depths from 0 to 6370 km every 10 km, given in metres. The grid goes
  through the core.

The file has no vph and no eta (the mesh has them), so the dataset has no
isotropic vs or vp, as 3d_seismic_REVEAL.

Depths that are kept
--------------------
The script keeps the 289 depths from 0 to MAX_DEPTH_KM = 2880 km. At
2890 km, the layer is in the outer core (VSV below 200 m/s) at every
latitude from -28 to 28 degrees and in the mantle at every other latitude.
The values below 2880 km mix the mantle and the core. The script checks that
no kept layer has a core value (VSV below CORE_VSV) and that the layer below
MAX_DEPTH_KM has one, so a changed source file fails the check.

The core-mantle boundary of the source is therefore not at one depth. The
depths of the grid are 6371 km minus the radius in the mesh of the model,
and the mesh is ellipsoidal: its surface and its core-mantle boundary are
closer to the centre at the poles than at the equator. The same shows near
the surface. The file name says that the nodes outside the mesh were
filled ("nan_filled"): at the poles the layers at 0, 10 and 20 km have the
same values, and at the equator no layer is a copy. gdrift puts each value
at the radius 6371 km minus its depth, which is its radius in the mesh. In
the spherical Earth of gdrift, a structure of RevealLO is thus deeper below
the surface at the poles than at the equator, by the flattening of the
model at that depth. The script converts the file as published and does
not correct this. The top layers are also kept as published: they hold
speeds of the crust, copied down where the node is outside the mesh.

Longitudes
----------
The source longitudes go from 0 to 360 degrees, and the resampling of
GLAD-M35 takes a grid from -180 to 180 degrees. The script reorders the
columns: the source columns from 180 to 359.5 degrees become -180 to -0.5,
and the columns from 0 to 180 degrees follow. The source column at 360
degrees is left out. It equals the column at 0 degrees to better than
SEAM_TOLERANCE (the largest difference is about 5e-10 m/s), which the script
checks. The -180 and +180 columns of the reordered grid are both the source
column at 180 degrees, so there is no step at the date line.

Output
------
One flat HDF5 file in the layout of the other gdrift seismic models:

- `coordinates`: (N, 3) Cartesian positions in metres, on the 65,341-point
  Fibonacci lattice of the other seismic models, repeated on each of the 289
  kept depths, shallowest layer first.
- `rho` in kg/m^3, and `vpv`, `vsh`, `vsv` in m/s.

The storage is that of GLAD-M35 (`write_file`): float32 fields, float64
coordinates, gzip level 4 after the shuffle filter, one chunk per depth
layer. The sha256 of the output is reproducible only with the same h5py,
HDF5 and zlib (see the docstring of scripts/convert_glad_m35.py).

Resampling
----------
Bilinear interpolation in (latitude, longitude) on each kept depth, with
the function of GLAD-M35 (`resample`). No interpolation is done in depth.
The 0.5 degree source has 55 km between nodes at the equator, and the
output lattice has about 88 km between points. The lattice samples the
source and does not average it, so lateral detail shorter than about 180 km
is not represented correctly. `resample` converts the fields named in its
SPEED_FIELDS from km/s to m/s. The source speeds are in m/s, so
`read_source` returns them in km/s.

The existing 3d_seismic_REVEAL dataset was resampled from another REVEAL
file with a Wendland kernel (scripts/convert_reveal_wendland.py), so the two
REVEAL datasets are not resampled by the same method.

Usage
-----
    python scripts/convert_reveal_lo.py                    # download, convert, check, report
    python scripts/convert_reveal_lo.py --source FILE.nc   # use a local copy of the Zenodo file
    python scripts/convert_reveal_lo.py --write            # also write the files and the manifest

With --write, the file is saved twice: as gdrift/data-sia/<dataset name>.h5
(the input of scripts/upload_to_s3.py) and as gdrift/data/<sha256>.h5 (where
the loader finds it before the upload). The manifest gets a new entry, or an
updated one, with the sha256, the file name and `etag: null` until the upload.
See docs/dataset-releases.md.
"""
import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

# Reuse the package functions, so that the sha256 is computed exactly as in
# the loader, and the GLAD-M35 script, so that this dataset is resampled,
# checked and written by the same code as the GLAD datasets.
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))
from gdrift.io import file_hash  # noqa: E402
import convert_glad_m35 as m35  # noqa: E402

DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

DATASET_NAME = "3d_seismic_RevealLO"

SOURCE_FILE = "Reveal_33s_with_LO_nan_filled.nc"
SOURCE_URL = f"https://zenodo.org/api/records/23646069/files/{SOURCE_FILE}/content"
# sha256 of the Zenodo file (record 23646069, checked on 2026-11-16). Its
# md5, as listed by Zenodo, is bc359fa768efb80da094328cc74ca8c1.
SOURCE_SHA256 = "6d63a429d53a5d120eacb07e1951e9c5b0923b4d1e10e0ca51813c392f582924"

# Names of the fields in the source and in the output. The output names and
# their order are those of 3d_seismic_REVEAL.
SOURCE_NAMES = {"rho": "RHO", "vpv": "VPV", "vsh": "VSH", "vsv": "VSV"}
OUTPUT_FIELDS = ("rho", "vpv", "vsh", "vsv")
SPEED_FIELDS = ("vpv", "vsh", "vsv")

# m35.resample converts from km/s to m/s the fields named in m35.SPEED_FIELDS.
# The RevealLO speeds must be among them, and the density must not.
assert set(SPEED_FIELDS) <= set(m35.SPEED_FIELDS) and "rho" not in m35.SPEED_FIELDS

# Deepest depth that is kept, in km (see "Depths that are kept")
MAX_DEPTH_KM = 2880.0
# A VSV below this value, in m/s, is a value of the outer core. The slowest
# kept VSV is 2311 m/s, and the core values at 2890 km are below 200 m/s.
CORE_VSV = 1000.0

# Latitude band where the layer below MAX_DEPTH_KM is in the core: it is in
# the core at every node with |latitude| <= CORE_LAT_MAX and in the mantle at
# every node with |latitude| >= MANTLE_LAT_MIN, in degrees. Written into the
# docstring and the `note` attribute; main() checks them.
CORE_LAT_MAX = 28.0
MANTLE_LAT_MIN = 28.5

# Largest allowed difference, in m/s and kg/m^3, between the source columns
# at 0 and 360 degrees, of which the script keeps only the first.
SEAM_TOLERANCE = 1e-6

# Plausible ranges of the kept source fields, in m/s for the speeds and
# kg/m^3 for the density. The lower bounds refuse a speed in km/s.
RANGES = {"vsv": (2000.0, 8000.0), "vsh": (2000.0, 8000.0), "vpv": (4000.0, 14500.0),
          "rho": (2000.0, 6000.0)}

AUTHORS = "Schiller, C. J., Keating, S., Afanasiev, M., Marty, P., & Fichtner, A."
TITLE = ("RevealLO: A Model Update to the REVEAL Global Full-Waveform Inversion Model with "
         "Long-Period Minor- and Major-Arc Surface-Wave and Multi-Orbit Body Wave Data")
CITATION = f"{AUTHORS} (2026). {TITLE} [Data set]. Zenodo."
# Citation in the manifest, in the style of most other entries
SOURCE = f"Schiller, C. J., et al. '{TITLE}.' Zenodo (2026)."
DOI = "10.5281/zenodo.23646069"

DESCRIPTION = (
    "RevealLO global full-waveform inversion model, an update of REVEAL with long-period minor- "
    "and major-arc surface waves and multi-orbit body waves (33 s), with the fields of REVEAL: "
    "vsv, vsh, vpv and density. From the 0.5 degree grid of the Zenodo record "
    "(doi:10.5281/zenodo.23646069), 289 depths every 10 km from 0 to 2880 km. The depths are "
    "6371 km minus the radius in the ellipsoidal mesh of the model, so a structure is deeper "
    "below the surface at the poles than at the equator (at 2890 km the source is in the core "
    "within 28 degrees of the equator and in the mantle elsewhere). The source has no vph and "
    "no eta, so there is no vs or vp."
)


def download_source(directory):
    """Download the Zenodo file into `directory` and return its path."""
    path = Path(directory) / SOURCE_FILE
    print(f"Downloading {SOURCE_URL} (5.3 GB)")
    with urllib.request.urlopen(SOURCE_URL, timeout=1800) as response, open(path, "wb") as out:
        shutil.copyfileobj(response, out)
    return path


def to_signed_longitude(lon, values):
    """Reorder a grid with longitudes 0..360 into a grid with longitudes -180..180.

    Parameters
    ----------
    lon : numpy.ndarray, shape (n_lon,)
        Increasing longitudes in degrees, evenly spaced, from 0 to 360 with
        both ends present and 180 among them.
    values : numpy.ndarray, shape (..., n_lon)
        Field values with longitude as the last axis.

    Returns
    -------
    lon_signed : numpy.ndarray, shape (n_lon,)
        Longitudes from -180 to 180 degrees.
    values_signed : numpy.ndarray, shape (..., n_lon)
        The columns in the order of `lon_signed`. The -180 and +180 columns
        are both the source column at 180 degrees; the source column at 360
        degrees is left out (the caller checks that it equals the column at
        0 degrees).
    """
    assert lon[0] == 0.0 and lon[-1] == 360.0, (lon[0], lon[-1])
    half = int(np.flatnonzero(lon == 180.0)[0])
    # Columns 180..359.5 become -180..-0.5; columns 0..180 follow unchanged
    west = np.arange(half, lon.size - 1)
    east = np.arange(0, half + 1)
    order = np.concatenate([west, east])
    lon_signed = np.concatenate([lon[west] - 360.0, lon[east]])
    return lon_signed, values[..., order]


def read_source(path):
    """Read the Zenodo file, check its grid and keep the mantle layers.

    Parameters
    ----------
    path : Path
        The Zenodo file. Its sha256 must equal SOURCE_SHA256.

    Returns
    -------
    depth_km : numpy.ndarray, shape (n_depth,)
        The kept depths in km, from 0 to MAX_DEPTH_KM, increasing.
    lat, lon : numpy.ndarray
        Grid axes in degrees, increasing, from -90 to 90 and -180 to 180.
    fields : dict of numpy.ndarray, shape (n_depth, n_lat, n_lon)
        The fields of OUTPUT_FIELDS on the kept depths, as float64. The
        speeds are in km/s (the unit `m35.resample` takes), the density in
        kg/m^3.
    core_layer_vsv : numpy.ndarray, shape (n_lat, n_lon)
        VSV in m/s of the source layer below MAX_DEPTH_KM, on the source
        longitudes 0 to 359.5. `core_latitudes` reads the core band from it.

    Raises
    ------
    RuntimeError
        If the sha256 of the file is not SOURCE_SHA256.
    AssertionError
        If the file does not have the expected dimensions and regular global
        grid, if a kept value is NaN or outside RANGES, if the columns at 0
        and 360 degrees differ, or if MAX_DEPTH_KM is not the deepest depth
        without a core value.
    """
    # file_hash returns "sha256:<hex>", the format of the manifest checks
    sha = file_hash(path).split(":", 1)[1]
    if sha != SOURCE_SHA256:
        raise RuntimeError(f"{path} has sha256 {sha}, expected {SOURCE_SHA256}")

    with h5py.File(path, "r") as f:
        depth_m = f["depth"][...].astype(np.float64)
        lat = f["latitude"][...].astype(np.float64)
        lon = f["longitude"][...].astype(np.float64)
        raw = {}
        for name, source_name in SOURCE_NAMES.items():
            dset = f[source_name]
            # The NetCDF dimensions are HDF5 dimension scales. The fields
            # must be (latitude, longitude, depth); the shape alone would
            # not show a swap of two axes of the same length.
            dims = tuple(dset.dims[k][0].name.lstrip("/") for k in range(3))
            assert dims == ("latitude", "longitude", "depth"), (source_name, dims)
            raw[name] = dset[...].astype(np.float64)

    # Depths in m: 0 and then every 10 km, increasing. MAX_DEPTH_KM must be
    # on the grid and must not be the last depth.
    depth_km = depth_m / 1e3
    assert depth_km[0] == 0.0 and np.allclose(np.diff(depth_km), 10.0, rtol=0, atol=1e-9), "depths"
    k_max = int(np.flatnonzero(np.isclose(depth_km, MAX_DEPTH_KM, rtol=0, atol=1e-9))[0])
    assert k_max + 1 < depth_km.size, "no layer below MAX_DEPTH_KM"

    # The bilinear interpolation assumes a regular, increasing grid that
    # covers the whole sphere
    assert np.allclose(np.diff(lat), lat[1] - lat[0]), "latitudes must be evenly spaced"
    assert np.allclose(np.diff(lon), lon[1] - lon[0]), "longitudes must be evenly spaced"
    assert lat[0] == -90.0 and lat[-1] == 90.0, (lat[0], lat[-1])
    for name, values in raw.items():
        assert values.shape == (lat.size, lon.size, depth_km.size), (name, values.shape)

    # The layer below MAX_DEPTH_KM must have a core value, and no kept
    # layer may have one (see "Depths that are kept")
    vsv = raw["vsv"]
    assert vsv[:, :, :k_max + 1].min() >= CORE_VSV, f"a layer down to {MAX_DEPTH_KM} km has a core value"
    core_layer_vsv = vsv[:, :-1, k_max + 1].copy()
    assert core_layer_vsv.min() < CORE_VSV, f"the layer below {MAX_DEPTH_KM} km has no core value"

    # Kept layers, with depth as the first axis, as m35.resample takes them
    depth_km = depth_km[:k_max + 1]
    fields = {name: np.ascontiguousarray(np.moveaxis(values[:, :, :k_max + 1], 2, 0))
              for name, values in raw.items()}
    del raw, vsv

    # Values must be finite and plausible for the crust and mantle in m/s
    # and kg/m^3
    for name, (low, high) in RANGES.items():
        values = fields[name]
        assert np.all(np.isfinite(values)), f"{name} contains NaN or inf"
        assert low < values.min() and values.max() < high, name

    # The column at 360 degrees is left out: it must equal the column at 0
    seam = max(np.abs(v[:, :, 0] - v[:, :, -1]).max() for v in fields.values())
    assert seam < SEAM_TOLERANCE, f"columns at 0 and 360 degrees differ by {seam}"
    print(f"Columns at 0 and 360 degrees differ by up to {seam:.1e}")
    lon, fields = _reorder_longitudes(lon, fields)

    # Report how deep the copied top layers reach (see "Depths that are
    # kept"); they are kept as published
    report_copied_top_layers(depth_km, lat, fields["vsv"])

    # The speeds are in m/s in the source; m35.resample takes km/s
    for name in SPEED_FIELDS:
        fields[name] = fields[name] / 1e3
    print(f"Kept {depth_km.size} depths from {depth_km[0]:g} to {depth_km[-1]:g} km")
    return depth_km, lat, lon, fields, core_layer_vsv


def _reorder_longitudes(lon, fields):
    """Apply `to_signed_longitude` to every field and return the new axis and fields."""
    out = {}
    for name, values in fields.items():
        lon_signed, out[name] = to_signed_longitude(lon, values)
    return lon_signed, out


def report_copied_top_layers(depth_km, lat, vsv):
    """Print how deep the top layers that are copies of the layer below reach.

    A node outside the mesh holds the value of the node below it, so in
    such a column the top layers are equal. For the equator and the two
    poles, the function prints the deepest layer that equals the layer below
    it, as a check of the ellipsoidal mesh (see the module docstring).

    Parameters
    ----------
    depth_km : numpy.ndarray, shape (n_depth,)
    lat : numpy.ndarray, shape (n_lat,)
    vsv : numpy.ndarray, shape (n_depth, n_lat, n_lon)
    """
    top = min(6, depth_km.size - 1)
    # same[k] is True where layer k equals layer k + 1
    same = vsv[:top] == vsv[1:top + 1]
    # Number of leading copies in each column: 0 if layer 0 differs from
    # layer 1. The first False along depth ends the run of copies.
    n_copies = np.argmin(np.concatenate([same, np.zeros((1,) + same.shape[1:], bool)]), axis=0)
    for la in (-90.0, 0.0, 90.0):
        i = int(np.argmin(np.abs(lat - la)))
        k = int(n_copies[i].max())
        if k == 0:
            print(f"  latitude {lat[i]:5.1f}: no top layer is a copy")
        else:
            print(f"  latitude {lat[i]:5.1f}: the layers from 0 to {depth_km[k - 1]:g} km are copies "
                  f"of the layer at {depth_km[k]:g} km at some nodes")


def core_latitudes(lat, core_layer_vsv):
    """Return the latitude band where the layer below MAX_DEPTH_KM is in the core.

    Parameters
    ----------
    lat : numpy.ndarray, shape (n_lat,)
        Latitudes in degrees.
    core_layer_vsv : numpy.ndarray, shape (n_lat, n_lon)
        VSV in m/s of the layer below MAX_DEPTH_KM.

    Returns
    -------
    core_lat_max : float
        Largest |latitude| of a row that is in the core at every node.
    mantle_lat_min : float
        Smallest |latitude| of a row that is in the mantle at every node.

    Raises
    ------
    AssertionError
        If a row has both core and mantle nodes, or if a core row is not
        nearer the equator than every mantle row.
    """
    core = core_layer_vsv < CORE_VSV
    all_core = core.all(axis=1)
    no_core = ~core.any(axis=1)
    assert np.all(all_core | no_core), "a row has core and mantle nodes"
    core_lat_max = float(np.abs(lat[all_core]).max())
    mantle_lat_min = float(np.abs(lat[no_core]).min())
    assert core_lat_max < mantle_lat_min, (core_lat_max, mantle_lat_min)
    return core_lat_max, mantle_lat_min


def file_attributes(n_depth):
    """Return the attributes of the RevealLO file, in the order they are written.

    A separate function, so that the test of the released file can compare
    its attributes with these.

    Parameters
    ----------
    n_depth : int
        Number of depth layers in the file.

    Returns
    -------
    dict
        Attribute names and values.
    """
    return {
        "comment": (
            f"Converted from {SOURCE_FILE} (Zenodo record 23646069). Resampled to Fibonacci "
            f"sphere ({m35.NUM_POINTS_PER_LAYER} points per layer) by bilinear interpolation in "
            f"latitude and longitude, on the {n_depth} source depths from 0 to "
            f"{MAX_DEPTH_KM:g} km. By Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)"
        ),
        "num_depth_levels": np.int64(n_depth),
        "num_points_per_level": np.int64(m35.NUM_POINTS_PER_LAYER),
        "velocity_units": "m/s",
        "density_units": "kg/m3",
        "model_reference": CITATION,
        "doi": DOI,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "note": (
            "The source depths are 6371 km minus the radius in the ellipsoidal mesh of the model, "
            "so a structure is deeper below the surface at the poles than at the equator. At "
            f"{MAX_DEPTH_KM + 10:g} km the source is in the core for |latitude| <= "
            f"{CORE_LAT_MAX:g} degrees and in the mantle for |latitude| >= {MANTLE_LAT_MIN:g} "
            "degrees, so the layers below "
            f"{MAX_DEPTH_KM:g} km are left out. Nodes outside the mesh hold the value of the node "
            "below them (at the poles, the layers at 0 and 10 km). The source has no vph and no eta."
        ),
    }


def build_file(path, coordinates, out, n_depth):
    """Write the HDF5 file of RevealLO with its attributes (see `m35.write_file`)."""
    m35.write_file(path, coordinates, out, file_attributes(n_depth))


def manifest_entry(sha):
    """Return the manifest entry for the new file.

    Parameters
    ----------
    sha : str
        sha256 of the output file (hex digits, without the "sha256:" prefix).

    Returns
    -------
    dict
        Entry for the "datasets" list of datasets.json, with `etag: None`
        until the file is uploaded (docs/dataset-releases.md).
    """
    return {
        "name": DATASET_NAME,
        "filename": f"{sha}.h5",
        "type": "TOMOGRAPHY_MODEL",
        "utility": "SEISMIC_MODEL",
        "source": SOURCE,
        "doi": DOI,
        "year": 2026,
        "fields": list(OUTPUT_FIELDS),
        "description": DESCRIPTION,
        "author": AUTHORS,
        "sha256": sha,
        "metadata_source": "web_search",
        "metadata_complete": True,
        "etag": None,
    }


def main():
    """Read the source, resample it, check it and write the file (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help=f"Local copy of {SOURCE_FILE}")
    parser.add_argument("--write", action="store_true", help="Write the files and update datasets.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = args.source if args.source else download_source(tmp)
        depth_km, lat, lon, fields, core_layer_vsv = read_source(source)
        print(f"Source: {depth_km.size} depths x {lat.size} lat x {lon.size} lon")

        # The latitudes of the core band are written into the note attribute;
        # a changed source must not leave them wrong
        band = core_latitudes(lat, core_layer_vsv)
        print(f"At {MAX_DEPTH_KM + 10:g} km: core for |lat| <= {band[0]:g}, mantle for |lat| >= {band[1]:g}")
        assert band == (CORE_LAT_MAX, MANTLE_LAT_MIN), band

        coordinates, out = m35.resample(depth_km, lat, lon, fields, output_fields=OUTPUT_FIELDS)
        m35.check_against_source(coordinates, out, depth_km, lat, fields)

        path = tmp / "out.h5"
        build_file(path, coordinates, out, depth_km.size)
        sha = file_hash(path).split(":", 1)[1]
        print(f"Output: {coordinates.shape[0]} points, {path.stat().st_size / 1e6:.0f} MB, sha256 {sha}")

        if not args.write:
            print("Dry run: nothing written. Use --write to save the files and update the manifest.")
            return

        DATA_SIA_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, DATA_SIA_DIR / f"{DATASET_NAME}.h5")
        shutil.copyfile(path, DATA_DIR / f"{sha}.h5")
        m35.update_manifest(manifest_entry(sha))
        print(f"Wrote gdrift/data-sia/{DATASET_NAME}.h5, gdrift/data/{sha}.h5 and the manifest entry")


if __name__ == "__main__":
    main()
