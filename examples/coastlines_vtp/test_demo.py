"""Regression tests for the coastlines_vtp demo.

The demo needs external GPlates data (rotation model + coastlines).
The fixture below resolves it in three steps, in order:

1. ``GDRIFT_DEMO_ROTATION_FILE`` / ``GDRIFT_DEMO_COASTLINES_FILE`` env
   vars if both point at existing files;
2. a local developer copy at the well-known ``gtrack/`` path;
3. download from the gadopt Digital Ocean Spaces CDN into a cache dir
   under ``gdrift/data-gplates/`` (so tests run in CI without setup).

Tests skip cleanly only when the optional deps (pygplates, pyvista,
mapbox_earcut) are not installed.
"""

import os
import pickle
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("pygplates")
pytest.importorskip("pyvista")
pytest.importorskip("mapbox_earcut")

DEMO_DIR = Path(__file__).parent

CDN_BASE = (
    "https://gadopt.syd1.cdn.digitaloceanspaces.com/g-drift/"
    "test_data/coastlines_vtp/"
)
ROTATION_NAME = "Muller_etal_2019_CombinedRotations.rot"
COASTLINES_NAME = "Muller_etal_2019_Global_Coastlines.gpmlz"

DEFAULT_GPLATES_ROOT = Path(
    "/Users/sghelichkhani/Workplace/gtrack/Zahirovic/"
    "Muller_etal_2019_v2_PlateMotionModel"
)


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(destination, "wb") as out:
        out.write(resp.read())


def _resolve_data_file(env_var, filename):
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        return Path(env_path)

    local = DEFAULT_GPLATES_ROOT / filename
    if local.exists():
        return local

    cache_root = Path(__file__).resolve().parents[2] / "gdrift" / "data-gplates"
    cached = cache_root / filename
    if not cached.exists():
        _download(CDN_BASE + filename, cached)
    return cached


ROTATION_FILE = _resolve_data_file("GDRIFT_DEMO_ROTATION_FILE", ROTATION_NAME)
COASTLINES_FILE = _resolve_data_file("GDRIFT_DEMO_COASTLINES_FILE", COASTLINES_NAME)

os.environ["GDRIFT_DEMO_ROTATION_FILE"] = str(ROTATION_FILE)
os.environ["GDRIFT_DEMO_COASTLINES_FILE"] = str(COASTLINES_FILE)


@pytest.fixture(scope="module")
def demo_namespace():
    """Run the demo script and capture its namespace."""
    namespace = {"__file__": str(DEMO_DIR / "demo.py")}
    cwd = os.getcwd()
    os.chdir(DEMO_DIR)
    try:
        exec(open(DEMO_DIR / "demo.py").read(), namespace)
    finally:
        os.chdir(cwd)
    return namespace


@pytest.fixture(scope="module")
def expected_values():
    with open(DEMO_DIR / "expected.pkl", "rb") as f:
        return pickle.load(f)


def test_all_ages_written(demo_namespace, expected_values):
    """One .vtp per requested age, all non-empty."""
    writer = demo_namespace["writer"]
    vtps = sorted(writer.vtp_dir.glob("*.vtp"))
    assert len(vtps) == expected_values["n_ages"]
    for vtp in vtps:
        assert vtp.stat().st_size > 0


def test_pvd_references_all_vtps(demo_namespace, expected_values):
    """The .pvd collection references exactly the generated .vtp files."""
    output_pvd = demo_namespace["output_pvd"]
    text = Path(output_pvd).read_text()
    for age in expected_values["ages"]:
        assert f"coastlines_{age:g}.vtp" in text


def test_present_day_mesh_matches(demo_namespace, expected_values):
    """Loading the age=0 VTP should give a consistent mesh size.

    The exact number of cells depends on pyvista's adaptive subdivision
    implementation, so we assert it is within a generous tolerance of
    the expected value rather than equal.
    """
    import pyvista as pv

    writer = demo_namespace["writer"]
    present_day = writer.vtp_dir / "coastlines_0.vtp"
    mesh = pv.read(present_day)
    expected_cells = expected_values["n_cells_age_0"]
    assert abs(mesh.n_cells - expected_cells) / expected_cells < 0.05


def test_surface_lives_on_sphere(demo_namespace):
    """Every vertex must lie on the unit sphere (earth_radius=1.0 in demo)."""
    import numpy as np
    import pyvista as pv

    writer = demo_namespace["writer"]
    mesh = pv.read(writer.vtp_dir / "coastlines_0.vtp")
    radii = np.linalg.norm(mesh.points, axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-6)
