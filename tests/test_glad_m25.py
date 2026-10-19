"""Tests for the GLAD-M25 dataset and scripts/convert_glad_m25.py.

Covers:
- the registry entry,
- the reading, the selection of the depths and the resampling of the
  conversion script, on three small synthetic files in the layout of the
  EarthScope EMC files (no network),
- the layers 10 m apart at a discontinuity, through EarthModel3D,
- the full Voigt averages that the script compares with vs and vp,
- the released file, only if it is already on disk (the test never downloads
  the 730 MB file).

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the conversion script is loaded from its path. The script puts ``scripts/``
on ``sys.path`` and imports scripts/convert_glad_m35.py, which the tests reach
as ``convert.m35``. tests/test_glad_m35.py loads its own module object of that
script; the two do not share state, so a monkeypatch of one does not change
the other.
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
from gdrift.earthmodel3d import EarthModel3D
from gdrift.utility import fibonacci_sphere, geodetic_to_cartesian

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_glad_m25.py"
NAME = "GLAD-M25"


def _load_script():
    """Import scripts/convert_glad_m25.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_glad_m25", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()


# ── Registry ────────────────────────────────────────────────────────────────

def test_glad_m25_is_registered_as_a_seismic_model():
    """The entry is a global seismic model with the seven fields."""
    ds = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}")
    assert ds is not None
    assert ds.dataset_type == DatasetType.TOMOGRAPHY_MODEL
    assert ds.utility == UtilityClass.SEISMIC_MODEL
    assert ds.fields == ["vs", "vsh", "vsv", "vp", "vph", "vpv", "eta"]
    assert ds.doi == "10.1093/gji/ggaa253"
    assert ds.regional is not True
    assert NAME in gdrift.AVAILABLE_SEISMIC_MODELS


# ── Full Voigt averages ─────────────────────────────────────────────────────

def test_full_voigt_averages_equal_the_simple_forms_for_eta_one():
    """With eta = 1 and vpv = vph, the full Voigt averages equal vs and vp.

    Then F = A - 2 L, and the shear modulus (A + C - 2 F + 5 N + 6 L) / 15
    reduces to (2 L + N) / 3, and kappa + 4 mu / 3 to A. Any vsv and vsh give
    zero difference; eta different from 1 does not.
    """
    vsv, vsh = np.array([4400.0, 4500.0]), np.array([4600.0, 4450.0])
    vph = np.array([8200.0, 8100.0])
    out = {"vsv": vsv, "vsh": vsh, "vpv": vph, "vph": vph, "eta": np.ones(2),
           "vs": convert.m35.voigt_vs(vsv, vsh), "vp": convert.m35.voigt_vp(vph, vph)}
    dvs, dvp = convert.full_voigt_difference(out)
    assert dvs < 1e-9 and dvp < 1e-9
    out["eta"] = np.full(2, 0.9)
    dvs, dvp = convert.full_voigt_difference(out)
    assert dvs > 1.0 and dvp > 1.0


def test_full_voigt_averages_match_the_stiffness_matrix():
    """The full Voigt averages equal those of the 6 x 6 stiffness matrix.

    For a transversely isotropic medium with a vertical symmetry axis (Voigt
    notation, density 1): C11 = C22 = A, C33 = C, C12 = A - 2 N,
    C13 = C23 = F, C44 = C55 = L, C66 = N. The Voigt bulk and shear moduli of
    any stiffness matrix are
    K = (C11 + C22 + C33 + 2 (C12 + C13 + C23)) / 9 and
    G = (C11 + C22 + C33 - (C12 + C13 + C23) + 3 (C44 + C55 + C66)) / 15.
    vs and vp are set to 0, so the returned differences are the full Voigt
    speeds themselves.
    """
    rng = np.random.default_rng(7)
    k = 50
    vsv, vsh = rng.uniform(4000, 5000, k), rng.uniform(4000, 5000, k)
    vpv, vph = rng.uniform(7500, 9000, k), rng.uniform(7500, 9000, k)
    eta = rng.uniform(0.85, 1.05, k)
    for i in range(k):
        a, c, l_, n_ = vph[i] ** 2, vpv[i] ** 2, vsv[i] ** 2, vsh[i] ** 2
        f_ = eta[i] * (a - 2 * l_)
        stiffness = np.zeros((6, 6))
        stiffness[:3, :3] = [[a, a - 2 * n_, f_], [a - 2 * n_, a, f_], [f_, f_, c]]
        stiffness[3, 3] = stiffness[4, 4] = l_
        stiffness[5, 5] = n_
        d, off = np.trace(stiffness[:3, :3]), stiffness[0, 1] + stiffness[0, 2] + stiffness[1, 2]
        bulk = (d + 2 * off) / 9
        shear = (d - off + 3 * np.trace(stiffness[3:, 3:])) / 15
        out = {"vsv": vsv[i:i + 1], "vsh": vsh[i:i + 1], "vpv": vpv[i:i + 1], "vph": vph[i:i + 1],
               "eta": eta[i:i + 1], "vs": np.zeros(1), "vp": np.zeros(1)}
        dvs, dvp = convert.full_voigt_difference(out)
        np.testing.assert_allclose([dvs, dvp], [np.sqrt(shear), np.sqrt(bulk + 4 * shear / 3)], rtol=1e-12)


# ── Conversion script, synthetic source ─────────────────────────────────────

# Depths of the synthetic source in km. The two layers above 6 km have zeros
# at some nodes, as in the published files, and 650 and 650.01 are the two
# sides of a discontinuity. float32, as in the source.
DEPTHS = np.array([-1.0, 5.0, 6.0, 100.0, 650.0, 650.01, 2891.0], dtype=np.float32)
KEPT = np.array([6.0, 100.0, 650.0, 650.01, 2891.0])
# Offset of each synthetic field, in source units (km/s; eta without unit)
OFFSETS = {"vsv": 4.4, "vsh": 4.6, "vpv": 8.0, "vph": 8.3, "eta": 0.95}
# Factor that multiplies the lateral plane (small for eta, so that it stays
# in a plausible range)
PLANE_SCALE = {"vsv": 1.0, "vsh": 1.0, "vpv": 1.0, "vph": 1.0, "eta": 1e-4}


def _plane(lat, lon):
    """A lateral pattern that bilinear interpolation reproduces exactly."""
    return 0.002 * lat + 0.0005 * lon + 1e-6 * lat * lon


def _expected(name, lat, lon, depth):
    """Source value of a synthetic field at (lat, lon, depth), in source units."""
    return PLANE_SCALE[name] * _plane(lat, lon) + OFFSETS[name] + 1e-5 * depth


def _write_sources(directory, zero_at=(-1.0, 5.0), lat_step=10.0):
    """Write three small HDF5 files with the names and layout of the EMC files.

    The longitude term is evaluated at -180 in the +180 column, so the two
    columns hold the same values, as for one meridian. Every field is 0 at a
    few nodes of the layers in `zero_at`.
    """
    lat = np.arange(-90.0, 90.0 + 1e-9, lat_step)
    lon = np.arange(-180.0, 180.0 + 1e-9, 10.0)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    lon2 = np.where(lon2 == 180.0, -180.0, lon2)
    for name, _, fields in convert.SOURCE_FILES.values():
        with h5py.File(Path(directory) / name, "w") as f:
            f["depth"] = DEPTHS
            f["latitude"] = lat.astype(np.float32)
            f["longitude"] = lon.astype(np.float32)
            for field in fields:
                values = np.stack([_expected(field, lat2, lon2, float(d)) for d in DEPTHS])
                for d in zero_at:
                    values[list(DEPTHS).index(np.float32(d)), 3:5, 7:9] = 0.0
                f[field] = values.astype(np.float32)


def _pin_sources(directory, monkeypatch):
    """Point SOURCE_FILES at the sha256 of the synthetic files."""
    pinned = {key: (name, gio.file_hash(Path(directory) / name).split(":", 1)[1], fields)
              for key, (name, _, fields) in convert.SOURCE_FILES.items()}
    monkeypatch.setattr(convert, "SOURCE_FILES", pinned)


def test_read_and_resample_synthetic_source(tmp_path, monkeypatch):
    """The kept depths, and every output field at the lattice points, are right."""
    _write_sources(tmp_path)

    with pytest.raises(RuntimeError, match="sha256"):
        convert.read_source(tmp_path)

    n = 300
    _pin_sources(tmp_path, monkeypatch)
    # The resampling, the checks and the writer are those of GLAD-M35 and
    # use its lattice size
    monkeypatch.setattr(convert.m35, "NUM_POINTS_PER_LAYER", n)
    d, lat, lon, fields = convert.read_source(tmp_path)
    # The layers above 6 km are left out, and 650.01 is rounded to 0.01 km
    np.testing.assert_array_equal(d, KEPT)
    coordinates, out = convert.m35.resample(d, lat, lon, fields, output_fields=convert.OUTPUT_FIELDS)
    assert list(out) == list(convert.OUTPUT_FIELDS)

    # Layers at the kept radii, shallowest first; 650 and 650.01 are 10 m apart
    radii = np.linalg.norm(coordinates, axis=1).reshape(KEPT.size, n)
    np.testing.assert_allclose(radii, (R_earth - KEPT * 1e3)[:, None] * np.ones((1, n)), rtol=0, atol=1e-6)

    # Latitude and longitude of the lattice computed here (z is the polar
    # axis). West of 170 E the last cell, whose +180 column holds the -180
    # values, is not used.
    unit = fibonacci_sphere(n)
    fib_lat = np.degrees(np.arcsin(unit[:, 2]))
    fib_lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    ok = fib_lon < 170.0

    def expected(name):
        """Expected output of a source field on the kept layers: m/s for speeds."""
        scale = 1.0 if name == "eta" else 1e3
        return np.stack([_expected(name, fib_lat, fib_lon, dk) for dk in KEPT]) * scale

    for name in OFFSETS:
        got = out[name].reshape(KEPT.size, n)
        np.testing.assert_allclose(got[:, ok], expected(name)[:, ok], rtol=1e-5, err_msg=name)

    # Isotropic speeds, with the formulas written out here
    vsv, vsh, vpv, vph = (expected(k) for k in ("vsv", "vsh", "vpv", "vph"))
    np.testing.assert_allclose(out["vs"].reshape(KEPT.size, n)[:, ok],
                               np.sqrt((2 * vsv**2 + vsh**2) / 3)[:, ok], rtol=1e-5)
    np.testing.assert_allclose(out["vp"].reshape(KEPT.size, n)[:, ok],
                               np.sqrt((vpv**2 + 4 * vph**2) / 5)[:, ok], rtol=1e-5)

    # The summary check passes, and the written file has float32 fields,
    # compression and the metadata of GLAD-M25
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


def test_layers_ten_metres_apart_stay_separate(tmp_path, monkeypatch):
    """EarthModel3D keeps 650 and 650.01 km as two layers, so a jump stays sharp.

    A quantity that is 0 down to 650 km and 1 from 650.01 km is 0.5 halfway
    between the two layers and 0 a few metres above 650 km.
    """
    _write_sources(tmp_path)
    _pin_sources(tmp_path, monkeypatch)
    n = 300
    monkeypatch.setattr(convert.m35, "NUM_POINTS_PER_LAYER", n)
    d, lat, lon, fields = convert.read_source(tmp_path)
    coordinates, _ = convert.m35.resample(d, lat, lon, fields, output_fields=convert.OUTPUT_FIELDS)

    # 300 points per layer are about 1300 km apart, so the default 200 km
    # limit would return NaN for every query
    model = EarthModel3D(default_max_distance=3000e3)
    model.set_coordinates(coordinates)
    model.add_quantity("step", np.repeat((KEPT >= 650.005).astype(float), n))
    model._detect_layer_structure()
    assert model._n_layers == KEPT.size and model._n_points_per_layer == n

    xyz = geodetic_to_cartesian(np.zeros(3), np.zeros(3), np.array([649.99e3, 650.005e3, 650.02e3]))
    np.testing.assert_allclose(model.at("step", xyz), [0.0, 0.5, 1.0], atol=1e-3)


def test_read_source_refuses_files_with_different_grids(tmp_path, monkeypatch):
    """A file whose latitudes differ from those of the vs file is refused."""
    _write_sources(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _write_sources(other, lat_step=5.0)
    eta = convert.SOURCE_FILES["eta"][0]
    (other / eta).replace(tmp_path / eta)
    _pin_sources(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="another grid"):
        convert.read_source(tmp_path)


@pytest.mark.parametrize("zero_at, message", [
    ((-1.0, 5.0, 100.0), "has a zero"),       # a zero in a kept layer
    ((-1.0,), "has no zero"),                 # MIN_DEPTH_KM is not the shallowest depth without zeros
])
def test_read_source_checks_min_depth(tmp_path, monkeypatch, zero_at, message):
    """MIN_DEPTH_KM must be the shallowest depth below which no layer has a zero."""
    _write_sources(tmp_path, zero_at=zero_at)
    _pin_sources(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match=message):
        convert.read_source(tmp_path)


def test_read_source_refuses_unphysical_speeds(tmp_path, monkeypatch):
    """A speed field in m/s (values above 15) is refused as a wrong unit."""
    _write_sources(tmp_path)
    with h5py.File(tmp_path / convert.SOURCE_FILES["vp"][0], "r+") as f:
        f["vph"][...] = f["vph"][...] * 1e3
    _pin_sources(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="vph"):
        convert.read_source(tmp_path)


def _assert_attributes(f, expected):
    """Check that an open HDF5 file has exactly the attributes in `expected`."""
    assert set(f.attrs) == set(expected)
    for key, value in expected.items():
        stored = f.attrs[key]
        assert stored == value, (key, stored, value)


def test_file_attributes_are_consistent():
    """The address, the DOIs, the source sha256 list and the Voigt numbers in the attributes are right."""
    attrs = convert.file_attributes(331)
    assert "siavash.ghelichkhan@anu.edu.au" in attrs["comment"]
    assert attrs["num_depth_levels"] == 331
    assert attrs["velocity_units"] == "m/s"
    assert attrs["doi"] == convert.DOI and attrs["repository_doi"] == convert.EMC_DOI
    assert attrs["source_sha256"] == ", ".join(f[1] for f in convert.SOURCE_FILES.values())
    assert f"{convert.VOIGT_DIFF_VS} m/s (vs)" in attrs["note"]
    assert f"{convert.VOIGT_DIFF_VP} m/s (vp)" in attrs["note"]


# ── Released file, only if it is already on disk ────────────────────────────

def _local_file():
    """Return the path of the released file if it is on disk, else None."""
    filename = DATASET_REGISTRY.get_dataset(f"3d_seismic_{NAME}").get_filename()
    for directory in (gio.DATA_PATH, gio.data_dir()):
        if (Path(directory) / filename).exists():
            return Path(directory) / filename
    return None


@pytest.mark.skipif(_local_file() is None, reason="the 730 MB file is not on disk; not downloaded by tests")
def test_released_glad_m25_file():
    """The released file loads as 331 layers with plausible values and the 650 km jump."""
    n = 65341
    # The attributes are those that the script writes
    with h5py.File(_local_file()) as f:
        _assert_attributes(f, convert.file_attributes(331))
    model = gdrift.SeismicModel(NAME)
    assert model.coordinates.shape == (331 * n, 3)
    fields = model.available_fields
    assert set(fields) == set(convert.OUTPUT_FIELDS)
    # The slowest values are the sediments at 6 km (see the script docstring)
    for name in ("vs", "vsh", "vsv"):
        assert 700 < fields[name].min() and fields[name].max() < 7600, name
    for name in ("vp", "vph", "vpv"):
        assert 1600 < fields[name].min() and fields[name].max() < 14000, name

    depths = (R_earth - np.linalg.norm(model.coordinates[::n], axis=1)) / 1e3
    np.testing.assert_allclose(depths[[0, -1]], [6.0, 2891.0], atol=1e-6)
    model._detect_layer_structure()
    assert model._n_layers == 331

    # Nearly isotropic below the upper mantle: vsh is within 0.1 m/s of vsv at 995 km
    layer = int(np.argmin(np.abs(depths - 995.0)))
    sl = slice(layer * n, (layer + 1) * n)
    assert np.abs(fields["vsh"][sl] - fields["vsv"][sl]).max() < 0.1

    # The mean of vs jumps by more than 300 m/s across the 10 m between the
    # 650 and 650.01 km layers
    above, below = (int(np.argmin(np.abs(depths - x))) for x in (650.0, 650.01))
    vs = fields["vs"].reshape(331, n)
    assert vs[below].mean() - vs[above].mean() > 300

    xyz = geodetic_to_cartesian(np.array([0.0, -45.0]), np.array([30.0, 120.0]), np.array([150e3, 2000e3]))
    v = model.at("vs", xyz)
    assert np.all(np.isfinite(v)) and np.all((v > 4000) & (v < 7500))
