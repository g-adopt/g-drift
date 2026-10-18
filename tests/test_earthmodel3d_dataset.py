"""Tests for 3D datasets that are not seismic tomography ("3d_earthmodel_").

Covers:
- the registry entries and AVAILABLE_EARTH_MODELS_3D,
- EarthModel3D.from_dataset and the loader that it shares with SeismicModel,
  on synthetic data (no network),
- the resampling in scripts/convert_lloyd_viscosity.py, on a synthetic NetCDF
  file (no network),
- the released Lloyd et al. (2024) file, only if it is already on disk
  (the test never downloads the 420 MB file).

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the conversion script is loaded from its path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

import gdrift
import gdrift.earthmodel3d as em3d
import gdrift.io as gio
import gdrift.seismic as seismic
from gdrift.constants import R_earth
from gdrift.datasetnames import DATASET_REGISTRY, DatasetType, UtilityClass
from gdrift.utility import fibonacci_sphere, geodetic_to_cartesian, regular_grid_bilinear_weights

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_lloyd_viscosity.py"
LLOYD = "Lloyd_et_al_2024_GJI"


def _load_script():
    """Import scripts/convert_lloyd_viscosity.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_lloyd_viscosity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()


# ── Registry ────────────────────────────────────────────────────────────────

def test_lloyd_dataset_is_registered_as_earth_model_3d():
    """The viscosity model is an EARTH_MODEL_3D dataset and not a seismic model."""
    ds = DATASET_REGISTRY.get_dataset(f"3d_earthmodel_{LLOYD}")
    assert ds is not None
    assert ds.dataset_type == DatasetType.EARTH_MODEL_3D
    assert ds.utility == UtilityClass.EARTH_MODEL_3D
    assert ds.fields == ["log10_viscosity"]
    assert ds.doi == "10.1093/gji/ggad455"
    assert LLOYD in gdrift.AVAILABLE_EARTH_MODELS_3D
    assert LLOYD not in gdrift.AVAILABLE_SEISMIC_MODELS


def test_available_earth_models_3d_matches_the_registry():
    """AVAILABLE_EARTH_MODELS_3D is the EARTH_MODEL_3D entries without their prefix."""
    expected = [ds.name.replace("3d_earthmodel_", "", 1)
                for ds in DATASET_REGISTRY.filter_by_type(DatasetType.EARTH_MODEL_3D)]
    assert gdrift.AVAILABLE_EARTH_MODELS_3D == expected
    assert len(expected) >= 1


# ── Loading, synthetic data ─────────────────────────────────────────────────

def _layered_dataset(n_points=500, depths_km=(0.0, 100.0, 400.0)):
    """Synthetic flat dataset in the layout of the gridded 3D files.

    The field is a linear function of depth, 20 + depth/100 (depth in km),
    so the radial interpolation between layers can be checked exactly.
    """
    unit = fibonacci_sphere(n_points)
    coords = np.vstack([unit * (R_earth - d * 1e3) for d in depths_km])
    field = np.concatenate([np.full(n_points, 20.0 + d / 100.0) for d in depths_km])
    return {"coordinates": coords, "log10_viscosity": field, "other": -field}


@pytest.fixture
def fake_registry(monkeypatch):
    """Serve the synthetic dataset for any name and record the requested names."""
    requested = []
    data = _layered_dataset()

    def fake_load(name, *args, **kwargs):
        requested.append(name)
        return {k: v.copy() for k, v in data.items()}

    monkeypatch.setattr(em3d, "load_dataset", fake_load)
    monkeypatch.setattr(em3d, "AVAILABLE_EARTH_MODELS_3D", ["Synthetic"])
    monkeypatch.setattr(seismic, "AVAILABLE_SEISMIC_MODELS", ["SynthSeis"])
    return requested


def test_from_dataset_loads_and_interpolates(fake_registry):
    """from_dataset uses the prefixed name, sets every field, and interpolates in depth."""
    # 500 points per layer are about 1000 km apart, so the distance beyond
    # which a query returns NaN must be larger than the 200 km default
    model = em3d.EarthModel3D.from_dataset("Synthetic", default_max_distance=3000e3)
    assert fake_registry == ["3d_earthmodel_Synthetic"]
    assert model.default_max_distance == 3000e3
    assert type(model) is em3d.EarthModel3D
    assert set(model.available_fields) == {"log10_viscosity", "other"}

    # Between 100 and 400 km the field is linear in depth, so any lateral
    # position gives exactly 20 + depth/100
    lat = np.array([10.0, -60.0, 85.0])
    lon = np.array([0.0, 179.9, -45.0])
    depth = np.array([150e3, 250e3, 399e3])
    value = model.at("log10_viscosity", geodetic_to_cartesian(lat, lon, depth))
    np.testing.assert_allclose(value, 20.0 + depth / 1e5, rtol=0, atol=1e-9)


def test_from_dataset_labels_and_errors(fake_registry):
    """Selected labels are loaded alone, the caller's list is kept, bad names raise."""
    labels = ["log10_viscosity"]
    model = em3d.EarthModel3D.from_dataset("Synthetic", labels=labels)
    assert list(model.available_fields) == ["log10_viscosity"]
    assert labels == ["log10_viscosity"]  # the coordinates are not appended to it

    with pytest.raises(ValueError, match="not present in dataset"):
        em3d.EarthModel3D.from_dataset("Synthetic", labels=["missing"])
    with pytest.raises(ValueError, match="not found in available 3D Earth models"):
        em3d.EarthModel3D.from_dataset("NoSuchModel")


def test_seismic_model_uses_the_shared_loader(fake_registry):
    """SeismicModel reads through the same loader, with the 3d_seismic_ prefix."""
    labels = ["log10_viscosity"]
    model = seismic.SeismicModel("SynthSeis", labels=labels)
    assert fake_registry[-1] == "3d_seismic_SynthSeis"
    assert list(model.available_fields) == ["log10_viscosity"]
    assert labels == ["log10_viscosity"]

    # On the subclass, from_dataset goes through the constructor
    model = seismic.SeismicModel.from_dataset("SynthSeis")
    assert isinstance(model, seismic.SeismicModel)
    assert fake_registry[-1] == "3d_seismic_SynthSeis"
    assert set(model.available_fields) == {"log10_viscosity", "other"}


# ── Conversion script, synthetic NetCDF ─────────────────────────────────────

def _plane(lat, lon):
    """A field that bilinear interpolation reproduces exactly."""
    return 20.0 + 0.01 * lat + 0.002 * lon + 1e-5 * lat * lon


def _write_netcdf(path, seam_depth_km=None):
    """Write a small NetCDF3 file with the layout of 3D_visc_infer_GLAD-M25.nc.

    The field is _plane(lat, lon) plus depth/1000 (depth in km). If
    seam_depth_km is given, the +180 column differs from the -180 column at
    that depth, as in the crust of the published file.
    """
    depth = np.array([0.0, 50.0, 250.0, 2891.0])
    lat = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    lon = np.arange(-180.0, 180.0 + 1e-9, 10.0)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    # The -180 and +180 columns must hold the same values, so the term in
    # longitude is evaluated at -180 for both ends
    lon_for_field = np.where(lon2 == 180.0, -180.0, lon2)
    field = np.stack([_plane(lat2, lon_for_field) + d / 1000.0 for d in depth])
    if seam_depth_km is not None:
        field[list(depth).index(seam_depth_km), :, -1] += 1.0
    with netcdf_file(path, "w") as nc:
        for name, values in (("depth", depth), ("latitude", lat), ("longitude", lon)):
            nc.createDimension(name, values.size)
            nc.createVariable(name, "f4", (name,))[:] = values
        nc.createVariable("log10eta", "f4", ("depth", "latitude", "longitude"))[:] = field
    return depth, lat, lon, field


def test_bilinear_weights_reproduce_a_bilinear_field():
    """The weights sum to 1, are exact at nodes and on the last grid lines."""
    lat = np.arange(-90.0, 90.1, 0.5)
    lon = np.arange(-180.0, 180.1, 0.5)
    rng = np.random.default_rng(3)
    q_lat = np.concatenate([rng.uniform(-90, 90, 1000), [-90.0, 90.0, 0.0]])
    q_lon = np.concatenate([rng.uniform(-180, 180, 1000), [-180.0, 180.0, 0.5]])
    i0, j0, w = regular_grid_bilinear_weights(lat, lon, q_lat, q_lon)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= -1e-12)

    grid = _plane(*np.meshgrid(lat, lon, indexing="ij"))
    corners = np.column_stack([grid[i0, j0], grid[i0 + 1, j0], grid[i0, j0 + 1], grid[i0 + 1, j0 + 1]])
    np.testing.assert_allclose(np.sum(corners * w, axis=1), _plane(q_lat, q_lon), atol=1e-10)


def test_read_and_resample_synthetic_source(tmp_path, monkeypatch):
    """read_source checks the hash and the grid; resample keeps depths and values."""
    path = tmp_path / "synthetic.nc"
    depth, lat, lon, field = _write_netcdf(path, seam_depth_km=50.0)

    # A file with another hash is refused
    with pytest.raises(RuntimeError, match="sha256"):
        convert.read_source(path)

    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    monkeypatch.setattr(convert, "NUM_POINTS_PER_LAYER", 400)
    d, la, lo, values = convert.read_source(path)
    np.testing.assert_array_equal(d, depth)

    coordinates, resampled = convert.resample(d, la, lo, values)
    assert coordinates.shape == (depth.size * 400, 3)
    # Layers are in source order, surface first, at the source radii
    radii = np.linalg.norm(coordinates, axis=1).reshape(depth.size, 400)
    np.testing.assert_allclose(radii, (R_earth - depth * 1e3)[:, None] * np.ones((1, 400)))

    # West of 170 E, each layer is _plane + depth/1000 at the lattice points.
    # The last cell (170 to 180 E) is excluded because the +180 column holds
    # the values of -180 there, and at 50 km also the extra seam offset.
    # Latitude and longitude of the lattice computed here from first
    # principles (z is the polar axis), not with gdrift.cartesian_to_geodetic,
    # so that a convention error in that function cannot cancel out
    unit = fibonacci_sphere(400)
    fib_lat = np.degrees(np.arcsin(unit[:, 2]))
    fib_lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    layers = resampled.reshape(depth.size, 400)
    ok = fib_lon < 170.0
    for k, dk in enumerate(depth):
        expected = _plane(fib_lat, fib_lon) + dk / 1000.0
        np.testing.assert_allclose(layers[k][ok], expected[ok], atol=1e-4)

    # The summary check passes on a correct resampling
    convert.check_against_source(coordinates, resampled, d, la, lo, values)


def test_each_side_of_the_seam_uses_its_own_column():
    """Just west of +180 the +180 column is used, just east of -180 the -180 column.

    The published file has different values in the two columns above 80 km,
    and the conversion keeps both, so each side must read only its own column.
    """
    lat = np.arange(-90.0, 90.1, 0.5)
    lon = np.arange(-180.0, 180.1, 0.5)
    grid = np.zeros((lat.size, lon.size))
    grid[:, 0] = 1.0   # the -180 column
    grid[:, -1] = 5.0  # the +180 column, different on purpose

    i0, j0, w = regular_grid_bilinear_weights(lat, lon, np.array([10.2, 10.2]), np.array([179.8, -179.8]))
    corners = np.column_stack([grid[i0, j0], grid[i0 + 1, j0], grid[i0, j0 + 1], grid[i0 + 1, j0 + 1]])
    value = np.sum(corners * w, axis=1)
    # 179.8 is 0.6 of the way from 179.5 to 180 (value 0 to 5); -179.8 is
    # 0.4 of the way from -180 to -179.5 (value 1 to 0)
    np.testing.assert_allclose(value, [0.6 * 5.0, 0.6 * 1.0], atol=1e-12)
    assert j0[0] == lon.size - 2 and j0[1] == 0


def test_read_source_refuses_a_seam_below_200_km(tmp_path, monkeypatch):
    """Columns at -180 and +180 that differ below 200 km stop the conversion."""
    path = tmp_path / "seam.nc"
    _write_netcdf(path, seam_depth_km=250.0)
    monkeypatch.setattr(convert, "SOURCE_SHA256", gio.file_hash(path).split(":", 1)[1])
    with pytest.raises(AssertionError, match="below 200 km"):
        convert.read_source(path)


# ── Released file, only if it is already on disk ────────────────────────────

def _local_lloyd_file():
    """Return the path of the released file if it is on disk, else None."""
    filename = DATASET_REGISTRY.get_dataset(f"3d_earthmodel_{LLOYD}").get_filename()
    for directory in (gio.DATA_PATH, gio.data_dir()):
        if (Path(directory) / filename).exists():
            return Path(directory) / filename
    return None


@pytest.mark.skipif(_local_lloyd_file() is None, reason="the 420 MB file is not on disk; not downloaded by tests")
def test_released_lloyd_file():
    """The released file loads, has the layered layout and the published values."""
    model = gdrift.EarthModel3D.from_dataset(LLOYD)
    values = model.available_fields["log10_viscosity"]
    assert model.coordinates.shape == (201 * 65341, 3)
    assert np.all(np.isfinite(values))
    # Published range: 1e18.26 to 1e39.08 Pa s; the crust is one constant value
    assert 18.2 < values.min() and values.max() < 39.1
    np.testing.assert_allclose(values[:65341], 26.178, atol=1e-3)

    # Mean of the 1000 km layer, and the step at 670 km of the published model
    radii = np.linalg.norm(model.coordinates[::65341], axis=1)
    layer = {round((R_earth - r) / 1e3): k for k, r in enumerate(radii)}
    mean = {d: values[layer[d] * 65341:(layer[d] + 1) * 65341].mean() for d in (650, 700, 1000)}
    assert abs(mean[1000] - 21.51) < 0.01
    assert mean[700] - mean[650] > 0.8

    # Interpolated values are finite in the mantle and NaN far above the surface
    xyz = geodetic_to_cartesian(np.array([-75.0, 30.0]), np.array([-100.0, 140.0]), np.array([200e3, 1500e3]))
    assert np.all(np.isfinite(model.at("log10_viscosity", xyz)))
    above = geodetic_to_cartesian(np.array([0.0]), np.array([0.0]), np.array([-300e3]))
    assert np.isnan(model.at("log10_viscosity", above))
