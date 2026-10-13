#!/usr/bin/env python3
"""Fix the units, the empty layers and the metadata of 21 seismic datasets.

The script reads each dataset as released in gdrift 0.1.4 (the files are
pinned by their sha256 in INPUT_SHA256), applies the fixes below in memory,
checks the result, and with --write saves the new files and updates
gdrift/datasets.json. A file that changes gets a new sha256 and therefore a new
file name on the server; the 0.1.4 files stay on the server unchanged.

Fixes (GitHub issue in brackets):

- Velocities in km/s [#36]. HMSL-S06, LLNL-G3Dv3, S362ANI+M, SAW24B16,
  SAW642AN, SAW642ANb and SEMum store absolute velocities in km/s. These
  fields are multiplied by 1000, so that every velocity in gdrift is in m/s.
- Empty layers [#37].
  PMEAN: the source has vph and vpv only from 50 to 700 km, while vp goes to
  2850 km. vph and vpv differ from vp only down to 225 km (the radial
  anisotropy of the PREM-like reference) and equal vp to within 0.001 m/s
  from 250 to 700 km, so below 700 km vph and vpv are set to vp.
  SEMUM2: the source has vs and vsh only down to 720 km, while vsv goes to
  1550 km. The model is not known to be isotropic below 720 km, so these
  layers stay NaN, and the depth range of each field is recorded in the
  attributes `depth_range_km_<field>`. Where the crust is thicker than 35 to
  55 km, the source gives vs = 0 and vsh = vsv = NaN; the conversion then
  mixed these zeros into vs. In the source, vs is the Voigt average of vsh
  and vsv, so vs is set to that average wherever it deviates from it by more
  than 0.1 %, and to NaN wherever vsh or vsv is NaN.
  SPani: the source stores equal-area blocks on a 5-degree grid, so most grid
  cells near the poles are empty. The points that the conversion left NaN
  (all within about 10 degrees of the poles) get the value of the source
  block they lie in: the valid cell of the nearest source latitude row that
  is nearest in longitude. This needs the GRD source collection
  (--grd-source-dir).
- Regional flag [#38]. SL2013NA covers North America only (84 % of each source
  layer is NaN). It gets the attribute regional = "true" and `regional: true`
  in the manifest, like the other regional models.
- Perturbation units [#39]. Every dvs and dvp field in gdrift is in percent,
  except LLNL-G3D-JPS, which was stored as a fraction. Its perturbations are
  multiplied by 100, and every file with a perturbation field gets the
  attribute perturbation_units = "percent".
- LLNL-G3D-JPS surface level. The level at 0 km of the source holds
  Vs = -2875.7 m/s and Vp = -1924.1 m/s at every point. These are not seismic
  speeds, so the level is removed. Queries above the next level (3.2 km, sea
  water) return the values of that level. Round-off values down to
  -2.2e-13 m/s in the sea-water level at 5.485 km are set to 0.
- HMSL-S06. Its source warns that vs was computed from dvs and ak135 and can be
  meaningless. The warning is added to the comment of the file.

Usage:
    python scripts/fix_seismic_datasets.py            # dry run: fix in memory, check, report
    python scripts/fix_seismic_datasets.py --write    # also write the files and the manifest

scripts/convert_seismic_models.py and scripts/convert_llnl_jps.py apply the
unit and regional fixes and the LLNL-G3D-JPS changes themselves. They do not
apply the PMEAN, SEMUM2 and SPani fixes or the HMSL-S06 warning, so run this
script on any dataset that they rebuild (with INPUT_SHA256 set to the
rebuilt files).

With --write, each fixed file is saved twice: as gdrift/data-sia/<dataset name>.h5
(the input of scripts/upload_to_s3.py) and as gdrift/data/<sha256>.h5 (where the
loader finds it before the upload). The manifest gets the new sha256 and
filename, and `etag: null` until the upload.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gdrift.io import _fetch, file_hash  # noqa: E402
from gdrift.datasetnames import hash_name  # noqa: E402

MANIFEST_PATH = REPO / "gdrift" / "datasets.json"

# GRD source collection, needed for the SPani block values
GRD_SOURCE_DIR = Path.home() / "Data" / "tomography"

# SPani field -> component directory of the GRD source collection
# (<GRD_SOURCE_DIR>/SPani_<component>_abs/<depth>_km.grd, values in km/s)
SPANI_COMPONENTS = {"vsv": "S_v", "vsh": "S_h", "vs": "S_i", "vpv": "P_v", "vph": "P_h", "vp": "P_i"}

# Largest relative deviation of SEMUM2 vs from the Voigt average of vsh and
# vsv that counts as round-off (the source agrees to about 1e-6)
VOIGT_TOLERANCE = 1e-3
DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

# Mean radius of the Earth in m, used by all seismic datasets to place the layers
R_EARTH = 6371e3

# Names of the absolute velocity fields (m/s after the fix) and of the
# perturbation fields (percent after the fix)
VELOCITY_FIELDS = ("vs", "vp", "vsh", "vsv", "vph", "vpv")
PERTURBATION_FIELDS = ("dvs", "dvp")

# The 0.1.4 files that this script starts from, by dataset name (without the
# "3d_seismic_" prefix) and sha256 of the content
INPUT_SHA256 = {
    "GAP": "e3174e81e0d47c89871d3439b06f6b7ec3919b6706bbcb77bd158430262a84ff",
    "HMSL-P06": "3dd028ac5b9eff70cb63a19e367d6ab168b3a61cb610bb56c5118bfabc2878cf",
    "HMSL-S06": "b3675e786ac3d6fd11ab203cffd86f778e91a109602bf268dca2a8ede6d39bd4",
    "LLNL-G3D-JPS": "935ce3edd822cabee6e726fe9e8b67a5bfc1f5fbaa9a85542c7dde606e79577f",
    "LLNL-G3Dv3": "4bd4674da226e858110eaa827f923cdbbecf82f39e0089ed4a35c5d362c9dc72",
    "MITP08": "43a9a286fe88aac4b33be26980ade6c68d3e1a0114898c59c815bcdda8cbda92",
    "OJP": "e90fafb45f82df65995f5bfdb8c28368304ef99e48d826db8dff7e337b9656d4",
    "PMEAN": "7549469bde0d2b94a442447719e83f4498f172d3a0bbe0e91b8d7a60adbefd58",
    "S362ANI+M": "58bd9d2948c62365227af109dbce0c51760b72cb51a8f490423b74fe78c37e7b",
    "SAW24B16": "92d32d48c94b670088ab7af7929d89dd9aafc52808a5d986e64f2760dd189480",
    "SAW642AN": "2b8e5c6e15977b1bb7b8bcbe903ebb9c32145770059d7f0e768a4202f1820462",
    "SAW642ANb": "a36af4245532ca2bfb3a15e75fa4ea83b0bba45f0ff6a72b8534cb96dd1355e3",
    "SEISGLOB2": "bd69efb22ddae5f26d6647356e70b1c2ff5817629aef453ce05995db88e1f628",
    "SEMUM2": "600e775e619576499bc5f939bd886a0f1a6795c6664e20837d42b6bc657084e7",
    "SEMum": "2487701d640f5d7c8aff48b326b255a4429e743f7ec4a9ded23770e72b2b4306",
    "SGLOBE-rani": "6256ac021d5e0189ba2f87271b7223bce39354d0f7d913a3a5b97df7a318fc25",
    "SL2013NA": "280764b233c7a0f7e6e807b4fb03216d36a2164a4090251663f0703aa30bb4cd",
    "SP12RTS": "30af4600a4000b204425c160791c68b4fb183313e7ad3a086f8264abc4576cd2",
    "SPani": "c7de72fb194ffe716ddc917a80ea6ef7da60d65e077ef0554b3451fb42015717",
    "TX2000": "9c2849ea71cbc2b91ebb3c83b0caca46108cf57d3d903b88ae3ce6a80eea1c00",
    "TX2019slab": "fd75b98eac2fed4f8f19f49a772cb753e3a4d8d25161eaed6a03c7fac8b6e205",
}

# Datasets with velocities in km/s (#36). The script stops if it finds km/s in
# any other dataset, or none in one of these.
KMS_DATASETS = {"HMSL-S06", "LLNL-G3Dv3", "S362ANI+M", "SAW24B16", "SAW642AN", "SAW642ANb", "SEMum"}

# Sentences added to the `comment` attribute of the fixed files
COMMENT_KMS = "Absolute velocities converted from km/s to m/s by gdrift."
COMMENT_PERCENT = "Velocity perturbations (dvs, dvp) are in percent."
COMMENT_PMEAN = ("Below 700 km, vph and vpv are set to vp by gdrift: the source has vph and vpv only"
                 " from 50 to 700 km, and they equal vp from 250 to 700 km.")
COMMENT_SEMUM2 = ("The source has vs and vsh only down to 720 km and vsv down to 1550 km. Below 720 km"
                  " vs and vsh are NaN. The attributes depth_range_km_<field> give the depth range of"
                  " each field. vs is the Voigt average of vsh and vsv, as in the source, and NaN where"
                  " vsh or vsv is NaN (inside thick crust at 35 to 55 km).")
COMMENT_SPANI = ("Points within about 10 degrees of the poles, where the source grid has no value, take"
                 " the value of the source block they lie in.")
COMMENT_JPS = ("The level at 0 km of the source (Vs = -2875.7 m/s, Vp = -1924.1 m/s at every point) is"
               " not included, so the top level is at 3.2 km (sea water: Vs = 0, Vp = 1500 m/s), and"
               " queries above it return its values. Perturbations are in percent, as in the source.")
COMMENT_HMSL_S06 = ("The source warns that vs was converted from dvs and ak135 and can be meaningless;"
                    " use it with care.")


# ── Functions on arrays (tested in tests/test_dataset_fixes.py) ──────────────

def layer_index(coordinates):
    """Group the points of a point cloud into layers of equal radius.

    Args:
        coordinates (ndarray): Cartesian coordinates in m, shape (n, 3).

    Returns:
        tuple: (radii, index). `radii` holds the distinct radii in m (rounded to
        1 mm, ascending); `index[i]` is the position in `radii` of point i.
    """
    r = np.round(np.linalg.norm(coordinates, axis=1), 3)
    radii, index = np.unique(r, return_inverse=True)
    return radii, index


def kms_to_ms(values):
    """Convert a velocity field from km/s to m/s.

    A field counts as km/s if its median is below 100. Mantle velocities are
    about 3 to 14 km/s, so a median below 100 can only be km/s, and a median
    above 100 can only be m/s.

    Args:
        values (ndarray): Velocity field, NaN allowed.

    Returns:
        tuple: (new values, True) for a km/s field, or (values, False) if the
        field is already in m/s.
    """
    if np.nanmedian(values) < 100.0:
        return values * 1000.0, True
    return values, False


def fill_empty_layers(target, source, index):
    """Copy `source` into the layers where `target` has no value at all.

    Used for PMEAN, where vph and vpv equal vp wherever the model is isotropic.
    Layers where `target` has at least one value are left as they are.

    Args:
        target (ndarray): Field with empty layers, shape (n,).
        source (ndarray): Field to copy from, same shape.
        index (ndarray): Layer of each point, from `layer_index`.

    Returns:
        tuple: (filled copy of target, number of layers filled).
    """
    out = target.copy()
    filled = 0
    for layer in np.unique(index):
        m = index == layer
        if np.all(np.isnan(target[m])) and np.any(np.isfinite(source[m])):
            out[m] = source[m]
            filled += 1
    return out, filled


def append_sentence(comment, sentence):
    """Append a sentence to a comment, ending the comment with a full stop first.

    Leading whitespace of the comment (some source comments start with a
    newline) is kept.

    Args:
        comment (str): Existing comment, possibly empty.
        sentence (str): Sentence to append.

    Returns:
        str: The combined comment.
    """
    body = comment.rstrip()
    if not body:
        return sentence
    if not body.endswith("."):
        body += "."
    return f"{body} {sentence}"


def voigt_vs(vsv, vsh):
    """Voigt-average isotropic S-wave speed sqrt((2 vsv^2 + vsh^2) / 3).

    With L = rho vsv^2 and N = rho vsh^2, the S-wave form of the Voigt
    average of the shear modulus is (2L + N) / 3. NaN in either input gives NaN.
    """
    return np.sqrt((2.0 * vsv**2 + vsh**2) / 3.0)


def restore_voigt_vs(vs, vsh, vsv, tolerance=VOIGT_TOLERANCE):
    """Make vs the Voigt average of vsh and vsv where it is not.

    Used for SEMUM2, whose source gives vs as the Voigt average of vsh and
    vsv but holds vs = 0 where vsh and vsv are NaN (inside thick crust). The
    conversion mixed these zeros into vs at neighbouring points.

    Args:
        vs, vsh, vsv (ndarray): Isotropic, horizontal and vertical S-wave speeds.
        tolerance (float): Relative deviation from the Voigt average that is
            accepted as round-off.

    Returns:
        tuple: (corrected copy of vs, number of points set to NaN because vsh
        or vsv is NaN, number of points set to the Voigt average).
    """
    out = vs.copy()
    average = voigt_vs(vsv, vsh)
    no_components = np.isfinite(vs) & ~np.isfinite(average)
    out[no_components] = np.nan
    both = np.isfinite(vs) & np.isfinite(average)
    off = np.zeros_like(both)
    off[both] = np.abs(vs[both] - average[both]) > tolerance * average[both]
    out[off] = average[off]
    return out, int(no_components.sum()), int(off.sum())


def block_values(lat, lon, grid_lat, grid_lon, grid):
    """Value of the source block that each point lies in.

    The source stores equal-area blocks on a regular grid: within a latitude
    row, each block is one valid cell and the cells around it are empty. A
    point belongs to the latitude row nearest to it, and within that row to
    the valid cell nearest in longitude (with wrap-around at 180 degrees).

    Args:
        lat, lon (ndarray): Point latitudes and longitudes in degrees.
        grid_lat (ndarray): Latitudes of the grid rows, shape (nlat,).
        grid_lon (ndarray): Longitudes of the grid columns in [-180, 180], shape (nlon,).
        grid (ndarray): Values, shape (nlat, nlon), NaN for empty cells.

    Returns:
        ndarray: Block value of each point, NaN if its row has no valid cell.
    """
    out = np.full(len(lat), np.nan)
    rows = np.abs(grid_lat[None, :] - np.asarray(lat)[:, None]).argmin(axis=1)
    for i, (row, point_lon) in enumerate(zip(rows, lon)):
        valid = np.isfinite(grid[row])
        if not valid.any():
            continue
        dlon = np.abs((grid_lon[valid] - point_lon + 180.0) % 360.0 - 180.0)
        out[i] = grid[row][valid][dlon.argmin()]
    return out


def read_grd_layer(path):
    """Read one GRD file of the source collection.

    Returns:
        tuple: (latitudes, longitudes in [-180, 180], values with NaN for
        empty cells), the values in the units of the file.
    """
    import netCDF4
    with netCDF4.Dataset(str(path)) as ds:
        names = ds.variables.keys()
        lon = np.asarray(ds.variables["lon" if "lon" in names else "x"][:], dtype=float)
        lat = np.asarray(ds.variables["lat" if "lat" in names else "y"][:], dtype=float)
        values = np.ma.filled(ds.variables["z"][:].astype(float), np.nan)
    lon[lon > 180.0] -= 360.0
    return lat, lon, values


def fill_spani_from_source(values, field, coordinates, radii, index, source_dir):
    """Fill the NaN points of one SPani field with the value of their source block.

    Args:
        values (ndarray): Field in m/s, shape (n,).
        field (str): Field name, a key of SPANI_COMPONENTS.
        coordinates (ndarray): Cartesian coordinates in m, shape (n, 3).
        radii, index (ndarray): Layers, from `layer_index`.
        source_dir (Path): GRD source collection.

    Returns:
        tuple: (filled copy of values, number of points filled).
    """
    out = values.copy()
    r = np.linalg.norm(coordinates, axis=1)
    lat = np.degrees(np.arcsin(coordinates[:, 2] / r))
    lon = np.degrees(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    directory = Path(source_dir) / f"SPani_{SPANI_COMPONENTS[field]}_abs"
    filled = 0
    for layer in np.unique(index[np.isnan(values)]):
        depth = (R_EARTH - radii[layer]) / 1e3
        path = directory / f"{depth:g}_km.grd"
        if not path.exists():
            raise FileNotFoundError(f"SPani source layer {path} not found")
        grid_lat, grid_lon, grid = read_grd_layer(path)
        bad = (index == layer) & np.isnan(values)
        # Source values are in km/s
        out[bad] = 1000.0 * block_values(lat[bad], lon[bad], grid_lat, grid_lon, grid)
        filled += int(np.isfinite(out[bad]).sum())
    return out, filled


def depth_range_km(values, radii, index):
    """Return the shallowest and deepest depth (km) at which a field has a value.

    Args:
        values (ndarray): Field, shape (n,).
        radii (ndarray): Radii of the layers in m, from `layer_index`.
        index (ndarray): Layer of each point, from `layer_index`.

    Returns:
        ndarray: [min depth, max depth] in km over the layers with at least one
        finite value.
    """
    layers = np.unique(index[np.isfinite(values)])
    depths = (R_EARTH - radii[layers]) / 1e3
    return np.array([depths.min(), depths.max()])


def clip_roundoff_negatives(values, tolerance=1e-6):
    """Set negative values that are only round-off (above -tolerance) to 0.

    A velocity cannot be negative. Values between -tolerance and 0 are
    round-off of 0 (for example in a sea-water layer, where Vs = 0). More
    negative values are left alone, so that `check_dataset` reports them.

    Args:
        values (ndarray): Velocity field in m/s.
        tolerance (float): Largest round-off in m/s.

    Returns:
        tuple: (corrected copy, number of values set to 0).
    """
    out = values.copy()
    small = (out < 0.0) & (out > -tolerance)
    out[small] = 0.0
    return out, int(small.sum())


def drop_surface_layer(coordinates, fields, max_depth_km=0.001):
    """Remove the points of the layer at the surface (depth at most `max_depth_km`).

    Args:
        coordinates (ndarray): Cartesian coordinates in m, shape (n, 3).
        fields (dict): Field name to array of shape (n,).
        max_depth_km (float): Points with a depth up to this value are removed.

    Returns:
        tuple: (coordinates, fields, number of points removed).
    """
    depth = (R_EARTH - np.linalg.norm(coordinates, axis=1)) / 1e3
    keep = depth > max_depth_km
    return coordinates[keep], {k: v[keep] for k, v in fields.items()}, int((~keep).sum())


# ── Applying the fixes to one dataset ────────────────────────────────────────

def fix_dataset(name, coordinates, fields, attrs, grd_source_dir=GRD_SOURCE_DIR):
    """Apply every fix that concerns dataset `name`.

    Args:
        name (str): Dataset name without the "3d_seismic_" prefix.
        coordinates (ndarray): Cartesian coordinates in m, shape (n, 3).
        fields (dict): Field name to array, in file order.
        attrs (dict): File attributes.
        grd_source_dir (Path): GRD source collection (only used for SPani).

    Returns:
        tuple: (coordinates, fields, attrs, report), where report is a list
        of strings that describe the changes.
    """
    fields = dict(fields)
    attrs = dict(attrs)
    report = []
    comment = str(attrs.get("comment", ""))

    # LLNL-G3D-JPS: remove the surface level before anything else
    if name == "LLNL-G3D-JPS":
        coordinates, fields, n = drop_surface_layer(coordinates, fields)
        report.append(f"removed the 0 km level ({n} points)")
        # The sea-water level at 5.485 km has Vs = 0 with round-off down to
        # -2.2e-13 m/s (-2.2e-16 km/s in the source). Set these to exactly 0.
        for key in ("vs", "vp"):
            fields[key], n = clip_roundoff_negatives(fields[key])
            if n:
                report.append(f"{key}: {n} round-off negatives set to 0")
        comment = append_sentence(comment, COMMENT_JPS)

    radii, index = layer_index(coordinates)

    # #36: velocities in km/s
    converted = []
    for key in VELOCITY_FIELDS:
        if key in fields:
            fields[key], changed = kms_to_ms(fields[key])
            if changed:
                converted.append(key)
    if bool(converted) != (name in KMS_DATASETS):
        raise RuntimeError(f"{name}: km/s fields {converted} do not match KMS_DATASETS")
    if converted:
        report.append(f"km/s -> m/s: {', '.join(converted)}")
        comment = append_sentence(comment, COMMENT_KMS)
    if any(k in fields for k in VELOCITY_FIELDS):
        attrs["velocity_units"] = "m/s"

    # #39: perturbations in percent
    if any(k in fields for k in PERTURBATION_FIELDS):
        if name == "LLNL-G3D-JPS":
            for key in PERTURBATION_FIELDS:
                fields[key] = fields[key] * 100.0
            report.append("dvs, dvp: fraction -> percent")
        else:
            comment = append_sentence(comment, COMMENT_PERCENT)
        attrs["perturbation_units"] = "percent"
        report.append("perturbation_units = percent")

    # #37: empty layers
    if name == "PMEAN":
        for key in ("vph", "vpv"):
            fields[key], n = fill_empty_layers(fields[key], fields["vp"], index)
            report.append(f"{key}: {n} empty layers set to vp")
        comment = append_sentence(comment, COMMENT_PMEAN)
    if name == "SEMUM2":
        fields["vs"], n_nan, n_avg = restore_voigt_vs(fields["vs"], fields["vsh"], fields["vsv"])
        report.append(f"vs: {n_nan} points without vsh/vsv set to NaN, {n_avg} set to the Voigt average")
        for key in ("vs", "vsh", "vsv"):
            attrs[f"depth_range_km_{key}"] = depth_range_km(fields[key], radii, index)
            report.append(f"depth_range_km_{key} = {attrs[f'depth_range_km_{key}'].tolist()}")
        comment = append_sentence(comment, COMMENT_SEMUM2)
    if name == "SPani":
        for key in list(fields):
            fields[key], n = fill_spani_from_source(fields[key], key, coordinates, radii, index, grd_source_dir)
            report.append(f"{key}: {n} NaN points filled from the source blocks")
        comment = append_sentence(comment, COMMENT_SPANI)

    # #38: regional flag
    if name == "SL2013NA":
        attrs["regional"] = "true"
        report.append("regional = true")

    if name == "HMSL-S06":
        comment = append_sentence(comment, COMMENT_HMSL_S06)

    attrs["comment"] = comment
    return coordinates, fields, attrs, report


def check_dataset(name, coordinates, fields, attrs):
    """Check a fixed dataset.

    Checks: no velocity field in km/s and none negative; every perturbation
    field marked as percent, with magnitudes that fit percent (99th
    percentile of |dv| above 0.5); no layer without any value unless a depth
    range is recorded for the field; vs equal to the Voigt average of vsh and
    vsv in SEMUM2 and SPani; no NaN left in SPani.

    Args:
        name (str): Dataset name without the "3d_seismic_" prefix.
        coordinates (ndarray): Cartesian coordinates in m, shape (n, 3).
        fields (dict): Field name to array.
        attrs (dict): File attributes.

    Returns:
        list of str: Problems found (empty if none).
    """
    problems = []
    radii, index = layer_index(coordinates)
    for key in VELOCITY_FIELDS:
        if key in fields:
            v = fields[key]
            if np.nanmedian(v) < 100.0:
                problems.append(f"{key} still in km/s")
            if np.nanmin(v) < 0.0:
                problems.append(f"{key} has negative values")
    for key in PERTURBATION_FIELDS:
        if key in fields:
            if attrs.get("perturbation_units") != "percent":
                problems.append(f"{key} without perturbation_units = percent")
            # A fraction would give a 99th percentile of |dv| of a few 0.01
            if np.nanpercentile(np.abs(fields[key]), 99) < 0.5:
                problems.append(f"{key} too small for percent")
    if name in ("SEMUM2", "SPani"):
        average = voigt_vs(fields["vsv"], fields["vsh"])
        both = np.isfinite(fields["vs"]) & np.isfinite(average)
        if np.any(np.abs(fields["vs"][both] - average[both]) > VOIGT_TOLERANCE * average[both]):
            problems.append("vs differs from the Voigt average of vsh and vsv")
        if np.any(np.isfinite(fields["vs"]) & ~np.isfinite(average)):
            problems.append("vs has values where vsh or vsv is NaN")
    # Empty layers are allowed only where a depth range is recorded
    for key, v in fields.items():
        for layer in np.unique(index):
            if np.all(np.isnan(v[index == layer])) and f"depth_range_km_{key}" not in attrs:
                problems.append(f"{key} empty at {(R_EARTH - radii[layer]) / 1e3:.1f} km")
                break
    if name == "SPani" and any(np.isnan(v).any() for v in fields.values()):
        problems.append("SPani still has NaN points")
    return problems


# ── Reading and writing files ────────────────────────────────────────────────

def read_input(name):
    """Fetch the 0.1.4 file of dataset `name` and return (coordinates, fields, attrs)."""
    sha = INPUT_SHA256[name]
    # A local copy can also carry the name-based file name of gdrift 0.1.3 and
    # earlier (the contents of these files did not change in 0.1.4)
    path = _fetch(sha + ".h5", "sha256:" + sha, "3d_seismic_" + name,
                  legacy_names=(hash_name("3d_seismic_" + name) + ".h5",))
    with h5py.File(path, "r") as f:
        coordinates = f["coordinates"][()]
        fields = {k: f[k][()] for k in f.keys() if k != "coordinates"}
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    return coordinates, fields, attrs


def write_output(path, coordinates, fields, attrs):
    """Write a dataset file in the layout of the input files (float64, no compression)."""
    with h5py.File(path, "w") as f:
        f.create_dataset("coordinates", data=coordinates)
        for key, values in fields.items():
            f.create_dataset(key, data=values)
        for key, value in attrs.items():
            f.attrs[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the fixed files and update datasets.json")
    parser.add_argument("--grd-source-dir", type=Path, default=GRD_SOURCE_DIR,
                        help="GRD source collection, for the SPani block values (default: ~/Data/tomography)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = {e["name"]: e for e in manifest["datasets"]}
    all_problems = {}
    outputs = {}

    for name in INPUT_SHA256:
        coordinates, fields, attrs = read_input(name)
        coordinates, fields, attrs, report = fix_dataset(name, coordinates, fields, attrs, args.grd_source_dir)
        problems = check_dataset(name, coordinates, fields, attrs)
        print(f"{name}: " + "; ".join(report))
        for p in problems:
            print(f"  PROBLEM {p}")
        if problems:
            all_problems[name] = problems
        outputs[name] = (coordinates, fields, attrs)

    if all_problems:
        sys.exit(f"Problems in {len(all_problems)} datasets; nothing written.")
    if not args.write:
        print("\nDRY RUN: pass --write to write the files and update datasets.json.")
        return

    DATA_SIA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, (coordinates, fields, attrs) in outputs.items():
        dataset = "3d_seismic_" + name
        path = DATA_SIA_DIR / f"{dataset}.h5"
        write_output(path, coordinates, fields, attrs)
        sha = file_hash(path).split(":", 1)[1]
        shutil.copyfile(path, DATA_DIR / f"{sha}.h5")
        entry = entries[dataset]
        entry["sha256"] = sha
        entry["filename"] = f"{sha}.h5"
        entry["etag"] = None
        if name == "SL2013NA":
            entry["regional"] = True
        print(f"{dataset}: {sha}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(outputs)} files and updated {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
