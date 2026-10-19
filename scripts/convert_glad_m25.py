#!/usr/bin/env python3
"""Convert the GLAD-M25 seismic tomography model to a gdrift dataset.

GLAD-M25 (Lei et al., 2020, GJI 223, 1-21, doi:10.1093/gji/ggaa253) is the
second generation of the Princeton global adjoint tomography models. It
starts from GLAD-M15 and fits waveforms of 1,480 earthquakes with 3-D
spectral-element simulations, in 10 quasi-Newton iterations. It is radially
anisotropic only in the upper mantle, with STW105 as the 1-D reference model,
and has no uncertainty estimates. GLAD-M35 (scripts/convert_glad_m35.py)
starts from it. The 3-D viscosity of Lloyd et al. (2024)
(scripts/convert_lloyd_viscosity.py) was inferred from its shear-wave
speeds, so this dataset is the seismic model that goes with that viscosity.

This script uses the resampling, the checks and the file writer of
scripts/convert_glad_m35.py, so that the two GLAD datasets are built in the
same way. Only the reading of the source, the selection of the depths and
the metadata are specific to GLAD-M25.

Source files
------------
EarthScope EMC (doi:10.17611/dp/emc.2020.gladm2500.1) distributes the model
sampled on a regular grid, in three files of data revision r0.0 (NetCDF4
classic, which h5py reads as HDF5):

- `glad-m25-vs.r0.0-n4c.nc`: vsv, vsh in km/s,
- `glad-m25-vp.r0.0-n4c.nc`: vpv, vph in km/s,
- `glad-m25-eta.r0.0-n4c.nc`: eta (dimensionless).

The script pins the sha256 of each file (SOURCE_FILES) and refuses any
other file. All three files have the same grid: latitudes -90 to 90 and
longitudes -180 to 180 every 0.5 degree, and 342 depths from -5 to 2891 km,
every 1 km to 20 km, every 2 km to 80 km, every 5 km to 800 km and every
15 km below, with 2891 km as the last depth. EMC calls the axis "depth
below earth surface". The negative depths, and the zeros in the deep ocean
at 5 km, show that the depths are measured from sea level, with negative
depths above it; the script uses radius = R_earth - depth. The depths are
float32 in the source, and the script rounds them to 0.01 km. This moves
410.01 and 650.01 by less than 1 cm (in float32 they are 410.0100098 and
650.0100098 km).

Depths that are kept
--------------------
Down to 5 km, the source has zeros in every field at the nodes that are
outside the solid Earth (in air above the topography, or in the ocean): 189
nodes at 5 km, and more than 100,000 nodes per layer from -5 to 2 km. A
zero is not a speed, and the bilinear interpolation would mix it with the
speeds of the nodes around it. The script therefore keeps only the depths
from MIN_DEPTH_KM = 6 km down, the shallowest depth without zeros. It checks
that no kept layer has a zero and that the layer above 6 km has one, so a
changed source file fails the check. The output has 331 depths, from 6 to
2891 km. At 6 km, the slowest nodes have vsv = 0.73 km/s and vpv = 1.67 km/s,
the speeds of the sediments of the crustal model; they are kept as
published. Through SeismicModel, a query above 6 km returns the values of
the 6 km layer, also over the ocean, up to `default_max_distance` (200 km)
above it, as for the other seismic models.

Two depths are given twice, 10 m apart: 410 and 410.01 km, and 650 and
650.01 km. EMC gives them for the two sides of the 410 and 650 km
discontinuities, but only the 650 km pair has the jump between its layers. At
650 km, the mean of vsv jumps from 5.56 to 5.93 km/s between the two layers.
At 410 km, the two layers are equal: the source puts the jump between 405 and
410 km (the mean of vsv goes from 4.81 to 5.00 km/s), so both 410 and 410.01
are below it. The script keeps both pairs as published. EarthModel3D starts
a new layer where the radius changes by more than 1 m, so each pair gives two
layers, and a query between them is linear in radius. Through SeismicModel,
the jump at 650 km therefore stays within 10 m, and the jump near 410 km is
spread over 405 to 410 km.

The 201 depths of the Lloyd et al. (2024) viscosity are not all among the
342 depths of this file: GLAD-M25 does not have the 30 odd depths of the
viscosity from 21 to 79 km, nor 28 of its depths every 50 km in the lower
mantle. Both datasets have the same lateral points, and gdrift interpolates
between depth layers, so the two can be queried at the same points.

Output
------
One flat HDF5 file in the layout of the other gdrift seismic models:

- `coordinates`: (N, 3) Cartesian positions in metres, on the 65,341-point
  Fibonacci lattice of the other seismic models, repeated on each of the 331
  kept depths, shallowest layer first.
- `vsv`, `vsh`, `vpv`, `vph` in m/s, `eta` without unit.
- `vs` and `vp`: isotropic speeds in m/s, computed from the resampled
  components with the same Voigt forms as GLAD-M35:

      vs = sqrt((2 vsv^2 + vsh^2) / 3)
      vp = sqrt((vpv^2 + 4 vph^2) / 5)

  The vp form equals the full Voigt average of the elastic tensor for
  eta = 1. The vs form equals it only for eta = 1 and vpv = vph, because the
  full Voigt shear modulus has the additional term (C - A) / 15 (see
  `full_voigt_difference`). With the eta, vpv and vph of GLAD-M25, the full
  Voigt averages differ from the forms by up to 29 m/s for vs and 50 m/s
  for vp (VOIGT_DIFF_VS and VOIGT_DIFF_VP, also in the `note`
  attribute). The forms are kept so that vs means the same in every gdrift
  seismic dataset. The model is nearly isotropic below 660 km: the largest
  difference between vsv and vsh is 46 m/s at 660 km, 3 m/s at 780 km and
  0.05 m/s at 995 km. From 650.01 km down, eta has one value in each layer.

The storage is that of GLAD-M35 (`write_file`): float32 fields, float64
coordinates, gzip level 4 after the shuffle filter, one chunk per depth
layer. The sha256 of the output is reproducible only with the same h5py,
HDF5 and zlib (see the docstring of scripts/convert_glad_m35.py). The
released file was written with h5py 3.16.0, HDF5 2.0.0 and zlib 1.2.12.

Resampling
----------
Bilinear interpolation in (latitude, longitude) on each kept depth, with
the function of GLAD-M35 (`resample`). No interpolation is done in depth.
The 0.5 degree source has 55 km between nodes at the equator, and the output
lattice has about 88 km between points. The lattice samples the source and
does not average it, so lateral detail shorter than about 180 km is not
represented correctly. The Lloyd et al. (2024) viscosity has the same 0.5
degree source grid and the same lattice, so both datasets lose the same
detail.

The columns at -180 and +180 longitude of the published file are not
identical. Below 6 km they differ by up to 0.91 km/s (in the crust), by up
to 0.39 km/s below 20 km, by up to 0.004 km/s below 40 km, and they are
identical from 85 km down. As for GLAD-M35, the -180 column is used for
longitudes just east of -180 and the +180 column for longitudes just west of
+180, so a step remains at the date line in the crust.

Usage
-----
    python scripts/convert_glad_m25.py                      # download, convert, check, report
    python scripts/convert_glad_m25.py --source-dir DIR     # use local copies of the three EMC files
    python scripts/convert_glad_m25.py --write              # also write the files and the manifest

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
# the loader, and the GLAD-M35 script, so that both GLAD datasets are
# resampled, checked and written by the same code.
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))
from gdrift.io import file_hash  # noqa: E402
import convert_glad_m35 as m35  # noqa: E402

DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

DATASET_NAME = "3d_seismic_GLAD-M25"

SOURCE_URL = "https://data.earthscope.org/archive/seismology/products/emc/netcdf/"
# The three EMC files: name, sha256 of the file as published (data revision
# r0.0, checked on 2026-10-19) and the fields read from it.
SOURCE_FILES = {
    "vs": ("glad-m25-vs.r0.0-n4c.nc",
           "8916f703b07b2c23587dc8f6a7eddd89505d5a5491bc6712cd9ca70fe1e23d93", ("vsv", "vsh")),
    "vp": ("glad-m25-vp.r0.0-n4c.nc",
           "8d0ac66d87eaecead0d1083c0bb6cd7c11d9caffa55e87328bdf484592cf2374", ("vpv", "vph")),
    "eta": ("glad-m25-eta.r0.0-n4c.nc",
            "1ce2905bf71c59d51d1377a4b7f5ab5137365ec917f4588236f8bc3d6ed05abc", ("eta",)),
}

# Shallowest depth that is kept, in km (see "Depths that are kept")
MIN_DEPTH_KM = 6.0

# Fields read from the source, in the order of the source files (read_source
# checks that SOURCE_FILES gives exactly these). The speeds are in km/s in
# the source and are stored in m/s; eta has no unit.
SPEED_FIELDS = ("vsv", "vsh", "vpv", "vph")
SOURCE_FIELDS = SPEED_FIELDS + ("eta",)
# Order of the fields in the output file and in the manifest
OUTPUT_FIELDS = ("vs", "vsh", "vsv", "vp", "vph", "vpv", "eta")

# m35.resample converts from km/s to m/s the fields named in m35.SPEED_FIELDS.
# The GLAD-M25 speeds must be among them, and eta must not.
assert set(SPEED_FIELDS) <= set(m35.SPEED_FIELDS) and "eta" not in m35.SPEED_FIELDS

# Plausible ranges of the kept source fields, in km/s for the speeds. The
# lower bounds allow the sediments at 6 km (vsv 0.73 km/s, vpv 1.67 km/s); the
# upper bounds refuse a field in m/s.
RANGES = {"vsv": (0.5, 8.0), "vsh": (0.5, 8.0), "vpv": (1.0, 15.0), "vph": (1.0, 15.0), "eta": (0.5, 1.5)}

# Largest differences, in m/s, between the full Voigt averages and the
# forms without eta in the output (computed by `full_voigt_difference`). They
# are written into the `note` attribute and the docstring.
VOIGT_DIFF_VS = 29
VOIGT_DIFF_VP = 50

AUTHORS = ("Lei, W., Ruan, Y., Bozdağ, E., Peter, D., Lefebvre, M., Komatitsch, D., Tromp, J., "
           "Hill, J., Podhorszki, N., & Pugmire, D.")
CITATION = (
    f"{AUTHORS} (2020). Global adjoint tomography - model GLAD-M25. "
    "Geophysical Journal International, 223(1), 1-21."
)
# Citation in the manifest, in the style of most other entries
SOURCE = (
    "Lei, W., et al. 'Global adjoint tomography - model GLAD-M25.' "
    "Geophysical Journal International 223.1 (2020): 1-21."
)
DOI = "10.1093/gji/ggaa253"
EMC_DOI = "10.17611/dp/emc.2020.gladm2500.1"

DESCRIPTION = (
    "GLAD-M25 global adjoint tomography model (predecessor of GLAD-M35, and the model from "
    "which the 3-D viscosity of Lloyd et al. (2024) was inferred): radially anisotropic in the "
    "upper mantle, without uncertainties. From the EarthScope EMC grid r0.0 "
    "(doi:10.17611/dp/emc.2020.gladm2500.1; 0.5 degree, 331 depths from 6 to 2891 km; the "
    "layers above 6 km, which have zeros outside the solid Earth, are left out; a query above "
    "6 km returns the 6 km values). The layers at "
    "650 and 650.01 km keep the jump there; near 410 km the jump lies between 405 and 410 km. "
    "vs and vp are "
    "sqrt((2 vsv^2 + vsh^2)/3) and sqrt((vpv^2 + 4 vph^2)/5), the Voigt forms without eta."
)


def download_source(directory):
    """Download the three EMC files into `directory` and return the directory."""
    for name, _, _ in SOURCE_FILES.values():
        url = SOURCE_URL + name
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=1800) as response, open(directory / name, "wb") as out:
            shutil.copyfileobj(response, out)
    return directory


def read_source(directory):
    """Read the three EMC files, check their grid and keep the solid layers.

    Parameters
    ----------
    directory : Path
        Directory with the three files of SOURCE_FILES. The sha256 of each
        must equal the one in SOURCE_FILES.

    Returns
    -------
    depth_km : numpy.ndarray, shape (n_depth,)
        The kept depths below sea level in km, from MIN_DEPTH_KM down,
        rounded to 0.01 km and increasing.
    lat, lon : numpy.ndarray
        Grid axes in degrees, increasing, from -90 to 90 and -180 to 180.
    fields : dict of numpy.ndarray, shape (n_depth, n_lat, n_lon)
        The source fields in SOURCE_FIELDS on the kept depths, as float64,
        in source units.

    Raises
    ------
    RuntimeError
        If the sha256 of a file is not the one in SOURCE_FILES.
    AssertionError
        If the files do not have the same regular global grid, if a kept
        layer has a zero or a value outside RANGES, or if MIN_DEPTH_KM is not
        the shallowest depth without zeros.
    """
    axes = None
    fields = {}
    for name, sha_expected, names in SOURCE_FILES.values():
        path = Path(directory) / name
        # file_hash returns "sha256:<hex>", the format of the manifest checks
        sha = file_hash(path).split(":", 1)[1]
        if sha != sha_expected:
            raise RuntimeError(f"{path} has sha256 {sha}, expected {sha_expected}")
        with h5py.File(path, "r") as f:
            file_axes = tuple(f[a][...] for a in ("depth", "latitude", "longitude"))
            # float32 in the source; float64 for the interpolation, as in
            # the other seismic datasets
            for field in names:
                fields[field] = f[field][...].astype(np.float64)
        # All three files must have the same grid, element by element
        if axes is None:
            axes = file_axes
        else:
            for a, b in zip(axes, file_axes):
                assert np.array_equal(a, b), f"{name} has another grid than {SOURCE_FILES['vs'][0]}"
    assert tuple(fields) == SOURCE_FIELDS, tuple(fields)

    depth_raw = axes[0].astype(np.float64)
    lat = axes[1].astype(np.float64)
    lon = axes[2].astype(np.float64)
    # The float32 depths 410.01 and 650.01 are 410.0100098 and 650.0100098.
    # Round to 0.01 km, the precision of the published depths.
    depth_km = np.round(depth_raw, 2)
    assert np.abs(depth_km - depth_raw).max() < 1e-4

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

    # Layers with a zero in any field have nodes outside the solid Earth.
    # MIN_DEPTH_KM must be the shallowest depth below which no layer has one.
    has_zero = np.zeros(depth_km.size, dtype=bool)
    for values in fields.values():
        has_zero |= np.any(values == 0.0, axis=(1, 2))
    keep = depth_km >= MIN_DEPTH_KM
    assert keep.any() and not has_zero[keep].any(), f"a layer from {MIN_DEPTH_KM} km down has a zero"
    above = np.flatnonzero(~keep)
    assert above.size and has_zero[above[-1]], f"the layer above {MIN_DEPTH_KM} km has no zero"
    print(f"Kept {keep.sum()} depths from {depth_km[keep][0]} to {depth_km[keep][-1]} km; "
          f"left out {above.size} depths from {depth_km[0]} to {depth_km[above[-1]]} km (zeros)")
    depth_km = depth_km[keep]
    fields = {name: values[keep] for name, values in fields.items()}

    # Values must be plausible for the crust and mantle in source units
    for name, (low, high) in RANGES.items():
        assert low < fields[name].min() and fields[name].max() < high, name

    # Report how much the columns at -180 and +180 differ (see the module
    # docstring); both are kept as published
    seam = max(np.abs(fields[n][:, :, 0] - fields[n][:, :, -1]).max() for n in SPEED_FIELDS)
    print(f"Columns at -180 and +180 differ in the speeds by up to {seam:.3f} km/s")
    return depth_km, lat, lon, fields


def full_voigt_difference(out):
    """Largest differences between the full Voigt averages and vs, vp of `out`.

    For a transversely isotropic medium with A = rho vph^2, C = rho vpv^2,
    L = rho vsv^2, N = rho vsh^2 and F = eta (A - 2 L), the Voigt averages of
    the bulk and shear moduli are

        kappa = (4 A + C + 4 F - 4 N) / 9,
        mu    = (A + C - 2 F + 5 N + 6 L) / 15.

    The density cancels in the speeds sqrt(mu / rho) and
    sqrt((kappa + 4 mu / 3) / rho), so it is set to 1.

    Parameters
    ----------
    out : dict of numpy.ndarray
        As returned by `m35.resample`, speeds in m/s.

    Returns
    -------
    dvs, dvp : float
        Largest absolute differences in m/s.
    """
    # Love parameters divided by the density, in (m/s)^2
    a_love = out["vph"].astype(np.float64) ** 2
    c_love = out["vpv"].astype(np.float64) ** 2
    l_love = out["vsv"].astype(np.float64) ** 2
    n_love = out["vsh"].astype(np.float64) ** 2
    f_love = out["eta"] * (a_love - 2.0 * l_love)
    kappa = (4.0 * a_love + c_love + 4.0 * f_love - 4.0 * n_love) / 9.0
    mu = (a_love + c_love - 2.0 * f_love + 5.0 * n_love + 6.0 * l_love) / 15.0
    dvs = np.abs(np.sqrt(mu) - out["vs"]).max()
    dvp = np.abs(np.sqrt(kappa + 4.0 * mu / 3.0) - out["vp"]).max()
    return float(dvs), float(dvp)


def file_attributes(n_depth):
    """Return the attributes of the GLAD-M25 file, in the order they are written.

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
            f"Converted from {', '.join(f[0] for f in SOURCE_FILES.values())} (EarthScope EMC, "
            f"data revision r0.0). Resampled to Fibonacci sphere ({m35.NUM_POINTS_PER_LAYER} points "
            f"per layer) by bilinear interpolation in latitude and longitude, on the {n_depth} source "
            f"depths from {MIN_DEPTH_KM:g} km down. vs and vp are the Voigt forms "
            "sqrt((2 vsv^2 + vsh^2)/3) and sqrt((vpv^2 + 4 vph^2)/5) of the resampled components. "
            "By Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)"
        ),
        "num_depth_levels": np.int64(n_depth),
        "num_points_per_level": np.int64(m35.NUM_POINTS_PER_LAYER),
        "velocity_units": "m/s",
        "model_reference": CITATION,
        "doi": DOI,
        "repository_doi": EMC_DOI,
        "source_file": ", ".join(f[0] for f in SOURCE_FILES.values()),
        "source_sha256": ", ".join(f[1] for f in SOURCE_FILES.values()),
        "note": (
            f"The source layers above {MIN_DEPTH_KM:g} km have zeros outside the solid Earth and "
            "are left out. Only the layers at 650 and 650.01 km have the jump of a discontinuity "
            "between them; the jump near 410 km lies between the 405 and 410 km layers. "
            "The vp form equals the full Voigt average for eta = 1, the vs form only for eta = 1 "
            "and vpv = vph; with the eta, vpv and vph of the model the full Voigt averages differ "
            f"by up to {VOIGT_DIFF_VS} m/s (vs) and {VOIGT_DIFF_VP} m/s (vp). eta has no unit."
        ),
    }


def build_file(path, coordinates, out, n_depth):
    """Write the HDF5 file of GLAD-M25 with its attributes (see `m35.write_file`)."""
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
        "year": 2020,
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
    parser.add_argument("--source-dir", type=Path, help="Directory with local copies of the three EMC files")
    parser.add_argument("--write", action="store_true", help="Write the files and update datasets.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = args.source_dir if args.source_dir else download_source(tmp)
        depth_km, lat, lon, fields = read_source(source)
        print(f"Source: {depth_km.size} depths x {lat.size} lat x {lon.size} lon")

        coordinates, out = m35.resample(depth_km, lat, lon, fields, output_fields=OUTPUT_FIELDS)
        m35.check_against_source(coordinates, out, depth_km, lat, fields)
        dvs, dvp = full_voigt_difference(out)
        print(f"Full Voigt averages differ from vs and vp by up to {dvs:.1f} and {dvp:.1f} m/s")
        # The rounded values are written into the note attribute; a changed
        # source must not leave them wrong
        assert round(dvs) == VOIGT_DIFF_VS and round(dvp) == VOIGT_DIFF_VP, (dvs, dvp)

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
