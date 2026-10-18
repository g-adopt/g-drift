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
from gdrift.utility import fibonacci_sphere, geodetic_to_cartesian

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
    """A field that bilinear interpolation reproduces exactly (km/s)."""
    return offset + 0.002 * lat + 0.0005 * lon + 1e-6 * lat * lon


def _write_source(path):
    """Write a small HDF5 file with the dataset names of the EMC file.

    Every field is _plane(lat, lon) plus a field-specific offset plus a
    depth term. The longitude term is evaluated at -180 in the +180 column,
    so the two columns hold the same values, as for one meridian.
    """
    depth = np.array([10.0, 100.0, 700.0, 2890.0])
    lat = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    lon = np.arange(-180.0, 180.0 + 1e-9, 10.0)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    lon2 = np.where(lon2 == 180.0, -180.0, lon2)
    offsets = {"vsv": 4.4, "vsh": 4.6, "vpv": 8.0, "vph": 8.3, "eta": 0.95, "std_vs": 0.02, "std_vp": 0.03}
    with h5py.File(path, "w") as f:
        f["depth"] = depth.astype(np.float32)
        f["latitude"] = lat.astype(np.float32)
        f["longitude"] = lon.astype(np.float32)
        for name, offset in offsets.items():
            scale = 1e-4 if name.startswith("std") or name == "eta" else 1.0
            values = np.stack([scale * _plane(lat2, lon2, 0.0) + offset + 1e-4 * d for d in depth])
            f[name] = values.astype(np.float32)
    return depth, offsets


def test_read_and_resample_synthetic_source(tmp_path, monkeypatch):
    """read_source checks the hash; resample converts units and computes the Voigt averages."""
    path = tmp_path / "synthetic.nc"
    depth, offsets = _write_source(path)

    with pytest.raises(RuntimeError, match="sha256"):
        convert.read_source(path)

    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    monkeypatch.setattr(convert, "NUM_POINTS_PER_LAYER", 300)
    d, lat, lon, fields = convert.read_source(path)
    coordinates, out = convert.resample(d, lat, lon, fields)
    assert list(out) == list(convert.OUTPUT_FIELDS)

    # Layers at the source radii, shallowest first
    radii = np.linalg.norm(coordinates, axis=1).reshape(depth.size, 300)
    np.testing.assert_allclose(radii, (R_earth - depth * 1e3)[:, None] * np.ones((1, 300)))

    # West of 170 E (away from the last cell, which holds the -180 values in
    # its +180 column) each field is the synthetic plane, in m/s for speeds.
    # Latitude and longitude are computed here, not with cartesian_to_geodetic.
    unit = fibonacci_sphere(300)
    fib_lat = np.degrees(np.arcsin(unit[:, 2]))
    fib_lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    ok = fib_lon < 170.0
    vsv = out["vsv"].reshape(depth.size, 300)
    expected = np.stack([_plane(fib_lat, fib_lon, 0.0) + offsets["vsv"] + 1e-4 * dk for dk in depth]) * 1e3
    np.testing.assert_allclose(vsv[:, ok], expected[:, ok], rtol=1e-6)
    # eta has no unit and is not scaled by 1000 like the speeds
    assert 0.9 < out["eta"].min() and out["eta"].max() < 2.0

    # The isotropic speeds are the Voigt averages of the stored components
    np.testing.assert_allclose(out["vs"], convert.voigt_vs(out["vsv"], out["vsh"]))
    np.testing.assert_allclose(out["vp"], convert.voigt_vp(out["vpv"], out["vph"]))

    # The summary check passes, and the written file has float32 fields
    convert.check_against_source(coordinates, out, d, lat, fields)
    out_path = tmp_path / "out.h5"
    convert.build_file(out_path, coordinates, out, depth.size)
    with h5py.File(out_path) as f:
        assert f["coordinates"].dtype == np.float64
        assert all(f[name].dtype == np.float32 for name in convert.OUTPUT_FIELDS)
        assert f.attrs["velocity_units"] == "m/s"


def test_read_source_refuses_unphysical_speeds(tmp_path, monkeypatch):
    """A speed field in m/s (values above 15) is refused as a wrong unit."""
    path = tmp_path / "bad.nc"
    _write_source(path)
    with h5py.File(path, "r+") as f:
        f["vsv"][...] = f["vsv"][...] * 1e3
    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    with pytest.raises(AssertionError, match="vsv"):
        convert.read_source(path)


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

    xyz = geodetic_to_cartesian(np.array([0.0, -45.0]), np.array([30.0, 120.0]), np.array([150e3, 2000e3]))
    vs = model.at("vs", xyz)
    assert np.all(np.isfinite(vs)) and np.all((vs > 4000) & (vs < 7500))
