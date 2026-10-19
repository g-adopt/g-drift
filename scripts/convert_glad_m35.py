#!/usr/bin/env python3
"""Convert the GLAD-M35 seismic tomography model to a gdrift dataset.

GLAD-M35 (Cui et al., 2024, GJI 239, 478-502, doi:10.1093/gji/ggae270) is the
third generation of the Princeton global adjoint tomography models. It starts
from GLAD-M25 and fits waveforms of 2,160 earthquakes (680 more than
GLAD-M25) with 3-D spectral-element simulations. It is a joint P and S model,
radially anisotropic only in the upper mantle, with STW105 as the 1-D
reference model. It is the first model of the series with uncertainty
estimates: the standard deviations of the P- and S-wave speeds come from a
low-rank approximation of the inverse Hessian built from the BFGS update
history. The article is open access under CC BY 4.0.

Source file
-----------
EarthScope EMC (doi:10.17611/dp/emc.2024.gladm35.1) distributes the model
sampled on a regular grid as `GLAD-M35.r0.1-n4c.nc` (NetCDF4 classic, data
revision r0.1, in which the topography of the 410 and 660 km discontinuities
was flattened). The script pins the sha256 of that file (SOURCE_SHA256) and
refuses any other file. NetCDF4 classic is an HDF5 file, so h5py reads it and
the script needs no netCDF library.

The file has, on 289 depths from 10 to 2890 km every 10 km, latitudes -90 to
90 and longitudes -180 to 180 every 1 degree:

- vpv, vph, vsv, vsh in km/s,
- eta (dimensionless),
- std_vp, std_vs: standard deviations of the P- and S-wave speeds, in km/s,
  as given by EMC. In the upper mantle the inversion parameters are vsv, vsh,
  the bulk sound speed and eta, so it is not stated which S-wave speed
  std_vs refers to there.

std_vp is exactly 0 at every node of the 10, 20, 30 and 40 km layers and of
the 2870, 2880 and 2890 km layers, where std_vs is not. There 0 marks
missing values. The script sets these layers of std_vp to NaN, the marker
for missing values in the other gdrift datasets. Through SeismicModel,
std_vp is therefore NaN shallower than 50 km and from 2860 km down (a
query exactly at 2860 km mixes in the NaN layer below it).

Output
------
One flat HDF5 file in the layout of the other gdrift seismic models:

- `coordinates`: (N, 3) Cartesian positions in metres, on the 65,341-point
  Fibonacci lattice of the other seismic models, repeated on each of the 289
  source depths, shallowest layer first.
- `vsv`, `vsh`, `vpv`, `vph`, `std_vs`, `std_vp` in m/s, `eta` without unit.
- `vs` and `vp`: isotropic speeds in m/s, computed from the resampled
  components:

      vs = sqrt((2 vsv^2 + vsh^2) / 3)
      vp = sqrt((vpv^2 + 4 vph^2) / 5)

  vs is the definition that the other gdrift seismic datasets use (see
  scripts/convert_seismic_models.py, `voigt_vs`). Both are the S- and P-wave
  forms of the Voigt average for a transversely isotropic medium used by
  Panning and Romanowicz (2006). They equal the full Voigt averages of the
  elastic tensor only for eta = 1 (and, for vs, vph = vpv). With the eta of
  GLAD-M35, the full Voigt averages differ from these forms by up to 29 m/s
  for vs and up to 54 m/s for vp; between 20 and 150 km depth the vp
  difference is systematic, about 30-40 m/s RMS (0.5 %). The simple forms
  are kept so that vs means the same in every gdrift seismic dataset. The
  paper does not define isotropic speeds. The model is isotropic
  (vsv = vsh, vpv = vph) from 780 km down; between 660 and 780 km the
  components still differ by up to 45 m/s. Computing vs and vp after the
  resampling makes them exactly the averages of the stored components.

Storage
-------
The fields are stored as float32, the precision of the source file; the
other seismic datasets store float64. The coordinates stay float64, because
EarthModel3D finds the depth layers from the differences of the radii
between consecutive points, which must be close to zero within a layer. The
arrays in `SeismicModel.available_fields` keep the stored float32 type;
`SeismicModel.at` returns float64.

All arrays are compressed losslessly with gzip (level 4) after the HDF5
shuffle filter. Without compression the file would be 1.13 GB, more than
twice the largest other dataset; with it, about 740 MB. h5py decompresses
on read, so the loader needs no change. Reading the whole file takes about
1.3 s instead of 0.1 s.

The compressed bytes depend on the HDF5 and zlib libraries, so the sha256 of
the output is reproducible only with the same h5py, HDF5 and zlib. The
released file was written with h5py 3.16.0, HDF5 2.0.0 and zlib 1.2.12. A
rebuild with other versions gives the same decoded arrays and attributes
but a different sha256; compare such a rebuild with the released file by
content, not by sha256. Any HDF5 version reads the file.

Resampling
----------
The source grid is regular, so the script uses bilinear interpolation in
(latitude, longitude) on each depth layer
(scripts/regular_grid.py, `regular_grid_bilinear_weights`). No interpolation is done in depth: the output
keeps the 289 source depths. The output lattice has about 88 km between
points. The 1 degree source has 111 km between nodes in latitude everywhere
and in longitude at the equator, so near the equator the lattice is finer
than the source. Towards the poles the source nodes get closer in longitude
than the lattice points, and some lateral detail there is smoothed.

The columns at -180 and +180 longitude of the published file are not
identical: they differ by more than 0.1 m/s in about a quarter of the
(depth, latitude) pairs, by up to 0.08 km/s in vsv (at 20 km depth). The
bilinear interpolation uses the -180 column for longitudes just east of -180
and the +180 column for longitudes just west of +180, so the file is
converted as published and a small step remains at the date line.

Use by scripts/convert_glad_m25.py
----------------------------------
The GLAD-M25 script imports this module and uses `resample`,
`check_against_source`, `write_file` and `update_manifest`, and through them
NUM_POINTS_PER_LAYER and SPEED_FIELDS. A change to any of these changes the
GLAD-M25 dataset too. SPEED_FIELDS must contain vsv, vsh, vpv and vph, and
must not contain eta; the GLAD-M25 script checks this when it is imported.

Usage
-----
    python scripts/convert_glad_m35.py                    # download, convert, check, report
    python scripts/convert_glad_m35.py --source FILE.nc   # use a local copy of the EMC file
    python scripts/convert_glad_m35.py --write            # also write the files and the manifest

With --write, the file is saved twice: as gdrift/data-sia/<dataset name>.h5
(the input of scripts/upload_to_s3.py) and as gdrift/data/<sha256>.h5 (where
the loader finds it before the upload). The manifest gets a new entry, or an
updated one, with the sha256, the file name and `etag: null` until the upload.
See docs/dataset-releases.md.
"""
import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

# Reuse the package functions, so that the coordinates and the sha256 are
# computed exactly as for the other datasets, and the bilinear weights shared
# by the conversion scripts (scripts/regular_grid.py), so that the
# interpolation is the same as for the Lloyd et al. (2024) viscosity.
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))
from gdrift.constants import R_earth  # noqa: E402
from gdrift.io import file_hash  # noqa: E402
from gdrift.utility import cartesian_to_geodetic, fibonacci_sphere  # noqa: E402
from regular_grid import regular_grid_bilinear_weights  # noqa: E402

MANIFEST_PATH = REPO / "gdrift" / "datasets.json"
DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

DATASET_NAME = "3d_seismic_GLAD-M35"

SOURCE_FILE = "GLAD-M35.r0.1-n4c.nc"
SOURCE_URL = f"https://data.earthscope.org/archive/seismology/products/emc/netcdf/{SOURCE_FILE}"
# sha256 of the EMC file as published (data revision r0.1, checked on 2026-10-18).
SOURCE_SHA256 = "07dc0c0b65c34996501334f9b0fd5d8576da435e04edbdfc66dea48d9e7d707b"

# Same lateral sampling as the other seismic models (scripts/convert_seismic_models.py).
NUM_POINTS_PER_LAYER = 65341

# Fields read from the source. Speeds and their standard deviations are in
# km/s in the source and are stored in m/s; eta has no unit.
SPEED_FIELDS = ("vsv", "vsh", "vpv", "vph", "std_vs", "std_vp")
SOURCE_FIELDS = SPEED_FIELDS + ("eta",)
# Order of the fields in the output file and in the manifest
OUTPUT_FIELDS = ("vs", "vsh", "vsv", "vp", "vph", "vpv", "eta", "std_vs", "std_vp")

AUTHORS = ("Cui, C., Lei, W., Liu, Q., Peter, D., Bozdağ, E., Tromp, J., Hill, J., "
           "Podhorszki, N., & Pugmire, D.")
CITATION = (
    f"{AUTHORS} (2024). GLAD-M35: a joint P and S global tomographic model with "
    "uncertainty quantification. Geophysical Journal International, 239(1), 478-502."
)
# Citation in the manifest, in the style of most other entries
SOURCE = (
    "Cui, C., et al. 'GLAD-M35: a joint P and S global tomographic model with "
    "uncertainty quantification.' Geophysical Journal International 239.1 (2024): 478-502."
)
DOI = "10.1093/gji/ggae270"
EMC_DOI = "10.17611/dp/emc.2024.gladm35.1"

DESCRIPTION = (
    "GLAD-M35 global adjoint tomography model (successor of GLAD-M25): joint P and S, "
    "radially anisotropic in the upper mantle, with standard deviations of Vs and Vp as "
    "given by EMC (std_vp is NaN at 10-40 km and 2870-2890 km, where the source has only "
    "zeros). From the EarthScope EMC grid r0.1 (doi:10.17611/dp/emc.2024.gladm35.1; 1 "
    "degree, 10-2890 km). vs and vp are sqrt((2 vsv^2 + vsh^2)/3) and "
    "sqrt((vpv^2 + 4 vph^2)/5), the Voigt forms without eta."
)


def download_source(directory):
    """Download the EMC file into `directory` and return its path."""
    path = directory / SOURCE_FILE
    print(f"Downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=600) as response, open(path, "wb") as out:
        shutil.copyfileobj(response, out)
    return path


def read_source(path):
    """Read the EMC file and check its grid.

    Parameters
    ----------
    path : Path
        The EMC file. Its sha256 must equal SOURCE_SHA256.

    Returns
    -------
    depth_km : numpy.ndarray, shape (n_depth,)
        Depths below the surface in km, increasing.
    lat, lon : numpy.ndarray
        Grid axes in degrees, increasing, from -90 to 90 and -180 to 180.
    fields : dict of numpy.ndarray, shape (n_depth, n_lat, n_lon)
        The source fields in SOURCE_FIELDS, as float64, in source units. The
        layers of std_vp that are 0 at every node are NaN.

    Raises
    ------
    RuntimeError
        If the sha256 of the file is not SOURCE_SHA256.
    AssertionError
        If the grid is not the regular global grid that the resampling expects.
    """
    # file_hash returns "sha256:<hex>", the format of the manifest checks
    sha = file_hash(path).split(":", 1)[1]
    if sha != SOURCE_SHA256:
        raise RuntimeError(f"{path} has sha256 {sha}, expected {SOURCE_SHA256}")

    with h5py.File(path, "r") as f:
        depth_km = f["depth"][...].astype(np.float64)
        lat = f["latitude"][...].astype(np.float64)
        lon = f["longitude"][...].astype(np.float64)
        # float32 in the source; float64 for the interpolation and the output,
        # as in the other seismic datasets
        fields = {name: f[name][...].astype(np.float64) for name in SOURCE_FIELDS}

    # The bilinear interpolation assumes a regular, increasing grid that
    # covers the whole sphere, with both -180 and +180 present
    shape = (depth_km.size, lat.size, lon.size)
    for name, values in fields.items():
        assert values.shape == shape, (name, values.shape)
        assert np.all(np.isfinite(values)), f"{name} contains NaN or inf"
    assert np.all(np.diff(depth_km) > 0), "depths must increase"
    assert np.allclose(np.diff(lat), lat[1] - lat[0]), "latitudes must be evenly spaced"
    assert np.allclose(np.diff(lon), lon[1] - lon[0]), "longitudes must be evenly spaced"
    assert lat[0] == -90.0 and lat[-1] == 90.0, (lat[0], lat[-1])
    assert lon[0] == -180.0 and lon[-1] == 180.0, (lon[0], lon[-1])
    # Speeds must be positive and plausible for the crust and mantle (km/s)
    for name in ("vsv", "vsh", "vpv", "vph"):
        assert 1.0 < fields[name].min() and fields[name].max() < 15.0, name
    assert fields["std_vs"].min() >= 0 and fields["std_vp"].min() >= 0

    # Layers of std_vp that are exactly 0 everywhere hold no value (see the
    # module docstring). Mark them as missing. Only whole layers are changed:
    # a zero at a single node would be kept.
    missing = np.all(fields["std_vp"] == 0.0, axis=(1, 2))
    fields["std_vp"][missing] = np.nan
    print(f"std_vp set to NaN at depths {depth_km[missing].tolist()} km (all zeros in the source)")

    # Report how much the columns at -180 and +180 differ (see the module
    # docstring); both are kept as published
    seam = np.abs(fields["vsv"][:, :, 0] - fields["vsv"][:, :, -1])
    print(f"Columns at -180 and +180 differ in vsv by up to {seam.max():.3f} km/s "
          f"(mean {seam.mean():.4f})")
    return depth_km, lat, lon, fields


def voigt_vs(vsv, vsh):
    """Isotropic S-wave speed sqrt((2 vsv^2 + vsh^2) / 3), the S-wave Voigt form.

    With L = rho vsv^2 and N = rho vsh^2, the Voigt average of the shear
    modulus over the S-wave moduli is (2 L + N) / 3. This is the definition of
    vs in the other gdrift seismic datasets.
    """
    return np.sqrt((2.0 * vsv**2 + vsh**2) / 3.0)


def voigt_vp(vpv, vph):
    """Isotropic P-wave speed sqrt((vpv^2 + 4 vph^2) / 5), the P-wave Voigt form.

    With C = rho vpv^2 and A = rho vph^2, this is the P-wave counterpart of
    `voigt_vs` used by Panning and Romanowicz (2006): it weights the
    horizontal P modulus four times the vertical one and leaves out eta.
    """
    return np.sqrt((vpv**2 + 4.0 * vph**2) / 5.0)


def resample(depth_km, lat, lon, fields, output_fields=None):
    """Resample the source grid onto the Fibonacci lattice of each depth.

    scripts/convert_glad_m25.py also uses this function, with its own source
    fields and output fields.

    Parameters
    ----------
    depth_km, lat, lon, fields
        As returned by `read_source`. Every field whose name is in
        SPEED_FIELDS is converted from km/s to m/s; the others keep their unit.
        The fields must include vsv, vsh, vpv and vph.
    output_fields : sequence of str, optional
        Names and order of the returned fields: any of the source fields and
        the isotropic speeds vs and vp. The default None means OUTPUT_FIELDS,
        read when the function is called.

    Returns
    -------
    coordinates : numpy.ndarray, shape (n_depth * NUM_POINTS_PER_LAYER, 3)
        Cartesian positions in metres, shallowest layer first.
    out : dict of numpy.ndarray, shape (n_depth * NUM_POINTS_PER_LAYER,)
        The fields of `output_fields`, speeds in m/s.
    """
    # Unit-sphere Fibonacci lattice, identical to the other seismic models
    unit = fibonacci_sphere(NUM_POINTS_PER_LAYER)
    # Latitude and longitude of each lattice point (z is the polar axis). The
    # depth output is meaningless for unit vectors and is discarded.
    fib_lat, fib_lon, _ = cartesian_to_geodetic(unit[:, 0], unit[:, 1], unit[:, 2])
    i0, j0, w = regular_grid_bilinear_weights(lat, lon, fib_lat, fib_lon)

    out = {}
    for name, values in fields.items():
        # Corner values for all layers at once: shape (n_depth, n_points, 4)
        corners = np.stack([values[:, i0, j0], values[:, i0 + 1, j0],
                            values[:, i0, j0 + 1], values[:, i0 + 1, j0 + 1]], axis=-1)
        resampled = np.einsum("dpk,pk->dp", corners, w).ravel()
        # km/s to m/s for speeds and their standard deviations
        out[name] = resampled * 1e3 if name in SPEED_FIELDS else resampled

    # Isotropic speeds from the resampled components (see the module docstring)
    out["vs"] = voigt_vs(out["vsv"], out["vsh"])
    out["vp"] = voigt_vp(out["vpv"], out["vph"])

    # Radius of each layer in metres; R_earth is the gdrift reference radius
    radii = R_earth - depth_km * 1e3
    coordinates = np.vstack([unit * r for r in radii])
    if output_fields is None:
        output_fields = OUTPUT_FIELDS
    return coordinates, {name: out[name] for name in output_fields}


def check_against_source(coordinates, out, depth_km, lat, fields):
    """Compare the resampled fields with the source and print a summary.

    Checks that the layer radii equal the source depths, that every resampled
    source field stays inside the range of its source layer (bilinear
    interpolation cannot overshoot), that vs and vp lie between their
    components, and prints the largest difference between the mean of each
    layer and the area-weighted (cos latitude) mean of the source layer.

    Parameters
    ----------
    coordinates, out
        As returned by `resample`.
    depth_km, lat, fields
        As returned by `read_source`. Every field in `fields` is checked, so
        scripts/convert_glad_m25.py can use this function for its own fields.

    Raises
    ------
    AssertionError
        If a radius or a value is outside the source.
    """
    n = NUM_POINTS_PER_LAYER
    n_depth = depth_km.size
    radii = np.linalg.norm(coordinates, axis=1).reshape(n_depth, n)
    assert np.allclose(radii, (R_earth - depth_km * 1e3)[:, None], rtol=0, atol=1e-3)

    # Cos-latitude weights for the area-weighted mean of a source layer
    wlat = np.cos(np.radians(lat))[:, None]
    worst = {}
    for name in fields:
        scale = 1e3 if name in SPEED_FIELDS else 1.0
        layers = out[name].reshape(n_depth, n) / scale
        src = fields[name]
        # Bounds over the full grid, both seam columns included, because the
        # interpolation uses both. Layers that are entirely NaN (std_vp) are
        # skipped, and must be entirely NaN in the output too.
        valid = ~np.all(np.isnan(src), axis=(1, 2))
        assert np.all(np.isnan(layers[~valid])), name
        assert not np.any(np.isnan(layers[valid])), name
        assert np.all(layers[valid].min(axis=1) >= src[valid].min(axis=(1, 2)) - 1e-9), name
        assert np.all(layers[valid].max(axis=1) <= src[valid].max(axis=(1, 2)) + 1e-9), name
        # For the mean, drop the +180 column so that the meridian is not
        # counted twice
        inner = src[valid][:, :, :-1]
        src_mean = np.sum(inner * wlat, axis=(1, 2)) / np.sum(wlat * np.ones(inner.shape[2]))
        worst[name] = np.abs(layers[valid].mean(axis=1) - src_mean).max()

    # Voigt averages lie between the two components they average
    for iso, (a, b) in (("vs", ("vsv", "vsh")), ("vp", ("vpv", "vph"))):
        low = np.minimum(out[a], out[b]) - 1e-6
        high = np.maximum(out[a], out[b]) + 1e-6
        assert np.all((out[iso] >= low) & (out[iso] <= high)), iso

    print("Largest difference of layer means (source units): "
          + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    for d in (10, 100, 410, 660, 1000, 2000, 2890):
        k = int(np.argmin(np.abs(depth_km - d)))
        vs = out["vs"].reshape(n_depth, n)[k]
        print(f"  {depth_km[k]:6.0f} km  vs {vs.min():7.1f} {vs.mean():7.1f} {vs.max():7.1f} m/s")


def build_file(path, coordinates, out, n_depth):
    """Write the HDF5 file of GLAD-M35 with its attributes (see `write_file`)."""
    attrs = {
        "comment": (
            f"Converted from {SOURCE_FILE} (EarthScope EMC, data revision r0.1). Resampled to "
            f"Fibonacci sphere ({NUM_POINTS_PER_LAYER} points per layer) by bilinear "
            "interpolation in latitude and longitude, on the 289 source depths. vs and vp are "
            "the Voigt averages sqrt((2 vsv^2 + vsh^2)/3) and sqrt((vpv^2 + 4 vph^2)/5) of the "
            "resampled components. By Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)"
        ),
        "num_depth_levels": np.int64(n_depth),
        "num_points_per_level": np.int64(NUM_POINTS_PER_LAYER),
        "velocity_units": "m/s",
        "model_reference": CITATION,
        "doi": DOI,
        "repository_doi": EMC_DOI,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "note": (
            "std_vs and std_vp are the standard deviations of the S- and P-wave speeds "
            "estimated by Cui et al. (2024) from the BFGS update history, as given by EMC, "
            "in m/s. std_vp is NaN at 10-40 km and 2870-2890 km, where the source has only "
            "zeros. vs and vp use the Voigt forms without eta (exact for eta = 1); with "
            "the eta of the model the full Voigt averages differ by up to 29 m/s (vs) and "
            "54 m/s (vp). eta has no unit."
        ),
    }
    write_file(path, coordinates, out, attrs)


def write_file(path, coordinates, out, attrs):
    """Write an HDF5 file in the layout of the other seismic models.

    scripts/convert_glad_m25.py also uses this function, so both GLAD files
    have the same storage (see "Storage" in the module docstring).

    Parameters
    ----------
    path : Path
        The output file. An existing file is overwritten.
    coordinates : numpy.ndarray, shape (N, 3)
        Cartesian positions in metres, stored as float64.
    out : dict of numpy.ndarray, shape (N,)
        The fields, stored as float32 in the order of the dict.
    attrs : dict
        Attributes of the file, written in the order of the dict.
    """
    # Lossless compression (see "Storage" in the module docstring). gzip and
    # shuffle are built into every HDF5 library, so any h5py can read the
    # file. Chunks of one depth layer keep the chunking deterministic, so the
    # sha256 of the output is reproducible with the same h5py, HDF5 and zlib
    # (see "Storage" for the versions of the released file).
    compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    layer = NUM_POINTS_PER_LAYER
    with h5py.File(path, "w") as f:
        f.create_dataset("coordinates", data=coordinates, chunks=(layer, 3), **compression)
        for name, values in out.items():
            # float32: the precision of the source (see the module docstring)
            f.create_dataset(name, data=values.astype(np.float32), chunks=(layer,), **compression)
        for key, value in attrs.items():
            f.attrs[key] = value


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
        "year": 2024,
        "fields": list(OUTPUT_FIELDS),
        "description": DESCRIPTION,
        "author": AUTHORS,
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
    parser.add_argument("--source", type=Path, help=f"Local copy of {SOURCE_FILE}")
    parser.add_argument("--write", action="store_true", help="Write the files and update datasets.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = args.source if args.source else download_source(tmp)
        depth_km, lat, lon, fields = read_source(source)
        print(f"Source: {depth_km.size} depths x {lat.size} lat x {lon.size} lon")

        coordinates, out = resample(depth_km, lat, lon, fields)
        check_against_source(coordinates, out, depth_km, lat, fields)

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
        update_manifest(manifest_entry(sha))
        print(f"Wrote gdrift/data-sia/{DATASET_NAME}.h5, gdrift/data/{sha}.h5 and the manifest entry")


if __name__ == "__main__":
    main()
