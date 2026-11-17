"""Tests for the RevealLO dataset and scripts/convert_reveal_lo.py.

Covers:
- the registry entry,
- the reordering of the longitudes from 0..360 to -180..180,
- the reading, the selection of the depths and the resampling of the
  conversion script, on a small synthetic file in the layout of the Zenodo
  file (no network),
- the checks of the depth cut at the core-mantle boundary, of the latitude
  band where the first left-out layer is in the core, of the seam at 0 and
  360 degrees, of the units and of the order of the dimensions,
- the released file, only if it is already on disk (the test never downloads
  the 540 MB file).

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the conversion script is loaded from its path. The script puts ``scripts/``
on ``sys.path`` and imports scripts/convert_glad_m35.py, which the tests reach
as ``convert.m35``. A monkeypatch of it is undone after each test.
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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_reveal_lo.py"
NAME = "RevealLO"


def _load_script():
    """Import scripts/convert_reveal_lo.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_reveal_lo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()


# ── Registry ────────────────────────────────────────────────────────────────

def test_reveal_lo_is_registered_as_a_seismic_model():
    """The entry is a global seismic model with the fields of REVEAL."""
    ds = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}")
    assert ds is not None
    assert ds.dataset_type == DatasetType.TOMOGRAPHY_MODEL
    assert ds.utility == UtilityClass.SEISMIC_MODEL
    assert ds.fields == ["rho", "vpv", "vsh", "vsv"]
    assert ds.fields == DATASET_REGISTRY.get_dataset("3d_seismic_REVEAL").fields
    assert ds.doi == "10.5281/zenodo.23646069"
    assert ds.regional is not True
    assert NAME in gdrift.AVAILABLE_SEISMIC_MODELS


# ── Longitudes ──────────────────────────────────────────────────────────────

def test_to_signed_longitude():
    """Columns 180..350 move to -180..-10, 0..180 follow, and 360 is left out."""
    lon = np.arange(0.0, 360.0 + 1e-9, 10.0)
    values = np.stack([lon, lon + 1000.0])  # two rows, longitude last
    lon_signed, out = convert.to_signed_longitude(lon, values)
    np.testing.assert_array_equal(lon_signed, np.arange(-180.0, 180.0 + 1e-9, 10.0))
    # Each column keeps its source value: the source longitude is lon_signed
    # mod 360, except at +180, which is the source column 180 too
    np.testing.assert_array_equal(out[0], np.mod(lon_signed, 360.0))
    np.testing.assert_array_equal(out[1], np.mod(lon_signed, 360.0) + 1000.0)
    # Both ends are the source column at 180 degrees
    assert out[0, 0] == out[0, -1] == 180.0


# ── Conversion script, synthetic source ─────────────────────────────────────

# Depths of the synthetic source in km: 0 to 3000 every 10 km, so the layer
# below MAX_DEPTH_KM (2880) exists, as in the 0 to 6370 km of the Zenodo file
DEPTHS_KM = np.arange(0.0, 3000.0 + 1e-9, 10.0)
KEPT = DEPTHS_KM[DEPTHS_KM <= convert.MAX_DEPTH_KM]
# Offset of each synthetic field, in source units (m/s, kg/m^3), and the
# factor of the lateral pattern
OFFSETS = {"vsv": 4400.0, "vsh": 4600.0, "vpv": 8000.0, "rho": 3300.0}
PLANE_SCALE = {"vsv": 1000.0, "vsh": 1000.0, "vpv": 1000.0, "rho": 100.0}
# |latitude| up to which the layer at 2890 km is in the core
CORE_LAT = 20.0


def _plane(lat, lon):
    """A lateral pattern that bilinear interpolation reproduces exactly (lon in -180..180)."""
    return 0.002 * lat + 0.0005 * lon + 1e-6 * lat * lon


def _expected(name, lat, lon, depth_km):
    """Source value of a synthetic field at (lat, lon, depth), in source units."""
    return PLANE_SCALE[name] * _plane(lat, lon) + OFFSETS[name] + 0.01 * depth_km


def _write_source(path, core_lat=CORE_LAT, seam_offset=0.0, km_per_s=(), swap_axes=False):
    """Write a small HDF5 file with the names and layout of the Zenodo file.

    The fields are (latitude, longitude, depth), with NetCDF dimension
    scales. Longitudes go from 0 to 360; the pattern is evaluated at the
    longitude in -180..180, so the columns at 0 and 360 hold the same values.
    Below MAX_DEPTH_KM, VSV is a core value (100 m/s) for |latitude| <=
    `core_lat`.

    Parameters
    ----------
    seam_offset : float
        Added to every field in the column at 360 degrees.
    km_per_s : tuple of str
        Speeds written in km/s instead of m/s.
    swap_axes : bool
        Write the fields as (longitude, latitude, depth).
    """
    lat = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    lon = np.arange(0.0, 360.0 + 1e-9, 10.0)
    lon_signed = np.where(lon > 180.0, lon - 360.0, lon)
    lat3, lon3, d3 = np.meshgrid(lat, lon_signed, DEPTHS_KM, indexing="ij")
    with h5py.File(path, "w") as f:
        for axis, values in (("latitude", lat), ("longitude", lon), ("depth", DEPTHS_KM * 1e3)):
            f[axis] = values
            f[axis].make_scale(axis)
        dims = ("longitude", "latitude", "depth") if swap_axes else ("latitude", "longitude", "depth")
        for name, source_name in convert.SOURCE_NAMES.items():
            values = _expected(name, lat3, lon3, d3)
            if name == "vsv":
                core = (np.abs(lat3) <= core_lat) & (d3 > convert.MAX_DEPTH_KM)
                values[core] = 100.0
            if name in km_per_s:
                values = values / 1e3
            values[:, -1, :] += seam_offset
            if swap_axes:
                values = np.swapaxes(values, 0, 1)
            dset = f.create_dataset(source_name, data=values)
            for k, axis in enumerate(dims):
                dset.dims[k].attach_scale(f[axis])
    return path


def _pin(path, monkeypatch):
    """Point SOURCE_SHA256 at the sha256 of the synthetic file."""
    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])


def test_read_and_resample_synthetic_source(tmp_path, monkeypatch):
    """The kept depths, the longitudes and every output field at the lattice points are right."""
    path = _write_source(tmp_path / "source.nc")

    with pytest.raises(RuntimeError, match="sha256"):
        convert.read_source(path)

    n = 300
    _pin(path, monkeypatch)
    # The resampling, the checks and the writer are those of GLAD-M35 and
    # use its lattice size
    monkeypatch.setattr(convert.m35, "NUM_POINTS_PER_LAYER", n)
    d, lat, lon, fields, core_layer_vsv = convert.read_source(path)
    # The layers below 2880 km are left out, and the longitudes are reordered
    np.testing.assert_array_equal(d, KEPT)
    np.testing.assert_array_equal(lon, np.arange(-180.0, 180.0 + 1e-9, 10.0))
    assert core_layer_vsv.shape == (lat.size, 36)
    # Speeds in km/s, as m35.resample takes them; density unchanged
    assert fields["vsv"].max() < 10.0 and fields["rho"].min() > 1000.0

    coordinates, out = convert.m35.resample(d, lat, lon, fields, output_fields=convert.OUTPUT_FIELDS)
    assert list(out) == list(convert.OUTPUT_FIELDS)

    # Layers at the kept radii, shallowest first
    radii = np.linalg.norm(coordinates, axis=1).reshape(KEPT.size, n)
    np.testing.assert_allclose(radii, (R_earth - KEPT * 1e3)[:, None] * np.ones((1, n)), rtol=0, atol=1e-6)

    # Latitude and longitude of the lattice computed here (z is the polar
    # axis). The cells next to +-180 degrees are not used: both end columns
    # hold the values at +180, so the plane is not exact at -180.
    unit = fibonacci_sphere(n)
    fib_lat = np.degrees(np.arcsin(unit[:, 2]))
    fib_lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    ok = np.abs(fib_lon) < 170.0

    for name in OFFSETS:
        # Output in m/s and kg/m^3, the units of the source
        expected = np.stack([_expected(name, fib_lat, fib_lon, dk) for dk in KEPT])
        got = out[name].reshape(KEPT.size, n)
        np.testing.assert_allclose(got[:, ok], expected[:, ok], rtol=1e-9, err_msg=name)

    # The summary check passes, and the written file has float32 fields,
    # compression and the metadata of RevealLO
    convert.m35.check_against_source(coordinates, out, d, lat, fields)
    out_path = tmp_path / "out.h5"
    convert.build_file(out_path, coordinates, out, KEPT.size)
    with h5py.File(out_path) as f:
        assert set(f) == {"coordinates", *convert.OUTPUT_FIELDS}
        assert f["coordinates"].dtype == np.float64
        assert all(f[name].dtype == np.float32 for name in convert.OUTPUT_FIELDS)
        assert all(f[name].compression == "gzip" for name in f)
        _assert_attributes(f, convert.file_attributes(KEPT.size))
        assert f.attrs["num_points_per_level"] == n


def test_core_latitudes(tmp_path, monkeypatch):
    """The core band of the first left-out layer is read from the source."""
    path = _write_source(tmp_path / "source.nc")
    _pin(path, monkeypatch)
    _, lat, _, _, core_layer_vsv = convert.read_source(path)
    assert convert.core_latitudes(lat, core_layer_vsv) == (CORE_LAT, CORE_LAT + 10.0)
    # A row with both core and mantle nodes is refused
    core_layer_vsv[0, :3] = 100.0
    with pytest.raises(AssertionError, match="core and mantle"):
        convert.core_latitudes(lat, core_layer_vsv)


def test_read_source_needs_a_core_value_below_the_cut(tmp_path, monkeypatch):
    """MAX_DEPTH_KM must be the deepest depth above the first layer with a core value."""
    # core_lat below 0: no node of the layer at 2890 km is in the core
    path = _write_source(tmp_path / "source.nc", core_lat=-1.0)
    _pin(path, monkeypatch)
    with pytest.raises(AssertionError, match="has no core value"):
        convert.read_source(path)


def test_read_source_refuses_a_core_value_in_a_kept_layer(tmp_path, monkeypatch):
    """A core value at 2880 km, the deepest kept layer, is refused."""
    path = _write_source(tmp_path / "source.nc")
    with h5py.File(path, "r+") as f:
        k = int(np.flatnonzero(DEPTHS_KM == convert.MAX_DEPTH_KM)[0])
        f["VSV"][5, 5, k] = 100.0
    _pin(path, monkeypatch)
    with pytest.raises(AssertionError, match="has a core value"):
        convert.read_source(path)


def test_read_source_refuses_a_seam(tmp_path, monkeypatch):
    """The column at 360 degrees is left out, so it must equal the column at 0."""
    path = _write_source(tmp_path / "source.nc", seam_offset=1.0)
    _pin(path, monkeypatch)
    with pytest.raises(AssertionError, match="0 and 360"):
        convert.read_source(path)


def test_read_source_refuses_speeds_in_km_per_s(tmp_path, monkeypatch):
    """A speed in km/s is below its range in m/s and is refused."""
    path = _write_source(tmp_path / "source.nc", km_per_s=("vpv",))
    _pin(path, monkeypatch)
    with pytest.raises(AssertionError, match="vpv"):
        convert.read_source(path)


def test_read_source_refuses_swapped_dimensions(tmp_path, monkeypatch):
    """Fields stored as (longitude, latitude, depth) are refused."""
    path = _write_source(tmp_path / "source.nc", swap_axes=True)
    _pin(path, monkeypatch)
    with pytest.raises(AssertionError, match="longitude"):
        convert.read_source(path)


def test_report_copied_top_layers(capsys):
    """The report names the copied top layers at the poles and none at the equator."""
    depth_km = np.arange(0.0, 100.0 + 1e-9, 10.0)
    lat = np.array([-90.0, 0.0, 90.0])
    rng = np.random.default_rng(3)
    vsv = rng.uniform(3000, 5000, (depth_km.size, 3, 4))
    # At the poles, the layers at 0 and 10 km hold the value at 20 km at one node
    vsv[0, 0, 1] = vsv[1, 0, 1] = vsv[2, 0, 1]
    vsv[0, 2, 3] = vsv[1, 2, 3] = vsv[2, 2, 3]
    convert.report_copied_top_layers(depth_km, lat, vsv)
    lines = capsys.readouterr().out.splitlines()
    assert "0 to 10 km are copies of the layer at 20 km" in lines[0]
    assert "no top layer is a copy" in lines[1]
    assert "0 to 10 km are copies of the layer at 20 km" in lines[2]


def _assert_attributes(f, expected):
    """Check that an open HDF5 file has exactly the attributes in `expected`."""
    assert set(f.attrs) == set(expected)
    for key, value in expected.items():
        stored = f.attrs[key]
        assert stored == value, (key, stored, value)


def test_file_attributes_are_consistent():
    """The address, the DOI, the source sha256, the units and the core band in the attributes are right."""
    attrs = convert.file_attributes(289)
    assert "siavash.ghelichkhan@anu.edu.au" in attrs["comment"]
    assert attrs["num_depth_levels"] == 289
    assert attrs["velocity_units"] == "m/s" and attrs["density_units"] == "kg/m3"
    assert attrs["doi"] == convert.DOI
    assert attrs["source_sha256"] == convert.SOURCE_SHA256
    assert f"|latitude| <= {convert.CORE_LAT_MAX:g} degrees" in attrs["note"]
    assert f"|latitude| >= {convert.MANTLE_LAT_MIN:g} degrees" in attrs["note"]


# ── Released file, only if it is already on disk ────────────────────────────

def _local_file():
    """Return the path of the released file if it is on disk, else None."""
    ds = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}")
    if ds is None:
        return None
    filename = ds.get_filename()
    for directory in (gio.DATA_PATH, gio.data_dir()):
        if (Path(directory) / filename).exists():
            return Path(directory) / filename
    return None


@pytest.mark.skipif(_local_file() is None, reason="the 540 MB file is not on disk; not downloaded by tests")
def test_released_reveal_lo_file():
    """The released file loads as 289 layers from 0 to 2880 km with plausible values."""
    n = 65341
    # The attributes are those that the script writes
    with h5py.File(_local_file()) as f:
        _assert_attributes(f, convert.file_attributes(289))
    model = gdrift.SeismicModel(NAME)
    assert model.coordinates.shape == (289 * n, 3)
    fields = model.available_fields
    assert set(fields) == set(convert.OUTPUT_FIELDS)
    # Ranges of the source on the kept layers (see RANGES)
    for name, (low, high) in convert.RANGES.items():
        assert low < fields[name].min() and fields[name].max() < high, name

    depths = (R_earth - np.linalg.norm(model.coordinates[::n], axis=1)) / 1e3
    np.testing.assert_allclose(depths[[0, -1]], [0.0, 2880.0], atol=1e-6)
    model._detect_layer_structure()
    assert model._n_layers == 289

    xyz = geodetic_to_cartesian(np.array([0.0, -45.0]), np.array([30.0, 120.0]), np.array([150e3, 2000e3]))
    v = model.at("vsv", xyz)
    assert np.all(np.isfinite(v)) and np.all((v > 4000) & (v < 7500))
