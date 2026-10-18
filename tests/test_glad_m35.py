"""Tests for the GLAD-M35 dataset and scripts/convert_glad_m35.py.

Covers:
- the registry entry,
- the Voigt averages and the resampling of the conversion script, on a small
  synthetic file in the layout of the EarthScope EMC file (no network),
- the released file, only if it is already on disk (the test never downloads
  the 1.1 GB file).

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the conversion script is loaded from its path.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

import gdrift
import gdrift.io as gio
from gdrift.constants import R_earth
from gdrift.datasetnames import DATASET_REGISTRY, DatasetType, UtilityClass
from gdrift.utility import fibonacci_sphere, geodetic_to_cartesian, regular_grid_bilinear_weights

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_glad_m35.py"
NAME = "GLAD-M35"


def _load_script():
    """Import scripts/convert_glad_m35.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_glad_m35", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()


# ── Registry ────────────────────────────────────────────────────────────────

def test_glad_m35_is_registered_as_a_seismic_model():
    """The entry is a global seismic model with all nine fields."""
    ds = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}")
    assert ds is not None
    assert ds.dataset_type == DatasetType.TOMOGRAPHY_MODEL
    assert ds.utility == UtilityClass.SEISMIC_MODEL
    assert ds.fields == ["vs", "vsh", "vsv", "vp", "vph", "vpv", "eta", "std_vs", "std_vp"]
    assert ds.doi == "10.1093/gji/ggae270"
    assert ds.regional is not True
    assert NAME in gdrift.AVAILABLE_SEISMIC_MODELS


# ── Voigt averages ──────────────────────────────────────────────────────────

def test_voigt_averages():
    """Equal components give that speed; the weights are 2:1 for S and 1:4 for P."""
    np.testing.assert_allclose(convert.voigt_vs(np.array([4500.0]), np.array([4500.0])), [4500.0])
    np.testing.assert_allclose(convert.voigt_vp(np.array([8000.0]), np.array([8000.0])), [8000.0])
    # vs^2 = (2 vsv^2 + vsh^2) / 3 and vp^2 = (vpv^2 + 4 vph^2) / 5
    np.testing.assert_allclose(convert.voigt_vs(np.array([3.0]), np.array([0.0])), [np.sqrt(6.0)])
    np.testing.assert_allclose(convert.voigt_vs(np.array([0.0]), np.array([3.0])), [np.sqrt(3.0)])
    np.testing.assert_allclose(convert.voigt_vp(np.array([5.0]), np.array([0.0])), [np.sqrt(5.0)])
    np.testing.assert_allclose(convert.voigt_vp(np.array([0.0]), np.array([5.0])), [np.sqrt(20.0)])


# ── Conversion script, synthetic source ─────────────────────────────────────

def _plane(lat, lon, offset):
    """A field that bilinear interpolation reproduces exactly."""
    return offset + 0.002 * lat + 0.0005 * lon + 1e-6 * lat * lon


# Offset of each synthetic field, in source units (km/s; eta without unit)
OFFSETS = {"vsv": 4.4, "vsh": 4.6, "vpv": 8.0, "vph": 8.3, "eta": 0.95, "std_vs": 0.02, "std_vp": 0.03}
# Factor that multiplies the lateral plane of each field (small for eta and
# the standard deviations, so that they stay in a plausible range)
PLANE_SCALE = {"vsv": 1.0, "vsh": 1.0, "vpv": 1.0, "vph": 1.0, "eta": 1e-4, "std_vs": 1e-4, "std_vp": 1e-4}
DEPTHS = np.array([10.0, 100.0, 700.0, 2890.0])


def _expected(name, lat, lon, depth):
    """Source value of a synthetic field at (lat, lon, depth), in source units."""
    return PLANE_SCALE[name] * _plane(lat, lon, 0.0) + OFFSETS[name] + 1e-5 * depth


def _write_source(path, zero_std_vp_at=None):
    """Write a small HDF5 file with the dataset names and layout of the EMC file.

    The longitude term is evaluated at -180 in the +180 column, so the two
    columns hold the same values, as for one meridian. If zero_std_vp_at is a
    depth, std_vp is 0 in that whole layer, as in the published file.
    """
    lat = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    lon = np.arange(-180.0, 180.0 + 1e-9, 10.0)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    lon2 = np.where(lon2 == 180.0, -180.0, lon2)
    with h5py.File(path, "w") as f:
        f["depth"] = DEPTHS.astype(np.float32)
        f["latitude"] = lat.astype(np.float32)
        f["longitude"] = lon.astype(np.float32)
        for name in OFFSETS:
            values = np.stack([_expected(name, lat2, lon2, d) for d in DEPTHS])
            if name == "std_vp" and zero_std_vp_at is not None:
                values[list(DEPTHS).index(zero_std_vp_at)] = 0.0
            f[name] = values.astype(np.float32)


def test_read_and_resample_synthetic_source(tmp_path, monkeypatch):
    """Every output field has the right value and unit at the lattice points."""
    path = tmp_path / "synthetic.nc"
    _write_source(path, zero_std_vp_at=10.0)

    with pytest.raises(RuntimeError, match="sha256"):
        convert.read_source(path)

    n = 300
    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    monkeypatch.setattr(convert, "NUM_POINTS_PER_LAYER", n)
    d, lat, lon, fields = convert.read_source(path)
    coordinates, out = convert.resample(d, lat, lon, fields)
    assert list(out) == list(convert.OUTPUT_FIELDS)

    # Layers at the source radii, shallowest first
    radii = np.linalg.norm(coordinates, axis=1).reshape(DEPTHS.size, n)
    np.testing.assert_allclose(radii, (R_earth - DEPTHS * 1e3)[:, None] * np.ones((1, n)))

    # Latitude and longitude of the lattice computed here (z is the polar
    # axis), not with cartesian_to_geodetic. West of 170 E the last cell,
    # whose +180 column holds the -180 values, is not used.
    unit = fibonacci_sphere(n)
    fib_lat = np.degrees(np.arcsin(unit[:, 2]))
    fib_lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    ok = fib_lon < 170.0

    def expected(name):
        """Expected output of a source field on all layers: m/s for speeds and std."""
        scale = 1.0 if name == "eta" else 1e3
        return np.stack([_expected(name, fib_lat, fib_lon, dk) for dk in DEPTHS]) * scale

    for name in OFFSETS:
        got = out[name].reshape(DEPTHS.size, n)
        want = expected(name)
        if name == "std_vp":
            # The all-zero layer at 10 km is missing data and becomes NaN
            assert np.all(np.isnan(got[0]))
            got, want = got[1:], want[1:]
        np.testing.assert_allclose(got[:, ok], want[:, ok], rtol=1e-5, err_msg=name)

    # Isotropic speeds, from the expected components with the formulas
    # written out here (not with the script's functions)
    vsv, vsh, vpv, vph = (expected(k) for k in ("vsv", "vsh", "vpv", "vph"))
    np.testing.assert_allclose(out["vs"].reshape(DEPTHS.size, n)[:, ok],
                               np.sqrt((2 * vsv**2 + vsh**2) / 3)[:, ok], rtol=1e-5)
    np.testing.assert_allclose(out["vp"].reshape(DEPTHS.size, n)[:, ok],
                               np.sqrt((vpv**2 + 4 * vph**2) / 5)[:, ok], rtol=1e-5)

    # The summary check passes, and the written file has float32 fields,
    # compression, and the NaN layer of std_vp
    convert.check_against_source(coordinates, out, d, lat, fields)
    out_path = tmp_path / "out.h5"
    convert.build_file(out_path, coordinates, out, DEPTHS.size)
    with h5py.File(out_path) as f:
        assert f["coordinates"].dtype == np.float64
        assert all(f[name].dtype == np.float32 for name in convert.OUTPUT_FIELDS)
        assert all(f[name].compression == "gzip" for name in f)
        assert f.attrs["velocity_units"] == "m/s"
        assert np.all(np.isnan(f["std_vp"][:n]))


def test_read_source_refuses_unphysical_speeds(tmp_path, monkeypatch):
    """A speed field in m/s (values above 15) is refused as a wrong unit."""
    path = tmp_path / "bad.nc"
    _write_source(path)
    with h5py.File(path, "r+") as f:
        f["vsv"][...] = f["vsv"][...] * 1e3
    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    with pytest.raises(AssertionError, match="vsv"):
        convert.read_source(path)


def test_bilinear_weights_refuse_points_outside_the_grid():
    """A 0 to 360 grid queried at a negative longitude raises instead of extrapolating."""
    lat = np.arange(-90.0, 90.1, 1.0)
    lon = np.arange(0.0, 360.1, 1.0)
    with pytest.raises(ValueError, match="outside the grid"):
        regular_grid_bilinear_weights(lat, lon, np.array([0.0]), np.array([-10.0]))
    # Points on the edges are inside
    _, _, w = regular_grid_bilinear_weights(lat, lon, np.array([-90.0, 90.0]), np.array([0.0, 360.0]))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert np.all((w >= 0) & (w <= 1))


# ── Released file, only if it is already on disk ────────────────────────────

def _local_file():
    """Return the path of the released file if it is on disk, else None."""
    filename = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}").get_filename()
    for directory in (gio.DATA_PATH, gio.data_dir()):
        if (Path(directory) / filename).exists():
            return Path(directory) / filename
    return None


@pytest.mark.skipif(_local_file() is None, reason="the 1.1 GB file is not on disk; not downloaded by tests")
def test_released_glad_m35_file():
    """The released file loads as a layered seismic model with plausible values."""
    model = gdrift.SeismicModel(NAME)
    assert model.coordinates.shape == (289 * 65341, 3)
    fields = model.available_fields
    assert set(fields) == set(convert.OUTPUT_FIELDS)
    for name in ("vs", "vsh", "vsv"):
        assert 2300 < fields[name].min() and fields[name].max() < 7600, name
    for name in ("vp", "vph", "vpv"):
        assert 4300 < fields[name].min() and fields[name].max() < 14000, name
    # Isotropic below the upper mantle: vsh equals vsv at 1000 km
    layer = int(np.argmin(np.abs(R_earth - np.linalg.norm(model.coordinates[::65341], axis=1) - 1000e3)))
    sl = slice(layer * 65341, (layer + 1) * 65341)
    np.testing.assert_allclose(fields["vsh"][sl], fields["vsv"][sl], rtol=1e-6)

    # Standard deviations in m/s; std_vp is NaN in the 10-40 and 2870-2890 km
    # layers, where the source has only zeros, and finite elsewhere
    depths = np.round((R_earth - np.linalg.norm(model.coordinates[::65341], axis=1)) / 1e3)
    std_vp = fields["std_vp"].reshape(289, 65341)
    missing = (depths <= 40) | (depths >= 2870)
    assert np.all(np.isnan(std_vp[missing])) and not np.any(np.isnan(std_vp[~missing]))
    assert 0 < np.nanmin(std_vp) and np.nanmax(std_vp) < 150
    assert 0 < fields["std_vs"].min() and fields["std_vs"].max() < 100

    xyz = geodetic_to_cartesian(np.array([0.0, -45.0]), np.array([30.0, 120.0]), np.array([150e3, 2000e3]))
    vs = model.at("vs", xyz)
    assert np.all(np.isfinite(vs)) and np.all((vs > 4000) & (vs < 7500))
