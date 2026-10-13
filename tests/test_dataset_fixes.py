"""Tests for the dataset fixes in scripts/convert_seismic_models.py and
scripts/fix_seismic_datasets.py.

``voigt_vs`` gives the Voigt-average isotropic S-wave speed, and
``fill_vs_from_voigt`` fills the NaN points of an isotropic ``vs`` from ``vsh``
and ``vsv`` (used for the GRD-collection models, SEMUCB-WM1 in particular).
``fix_mitp08_vp`` rebuilds the MITP08 P-wave speed with the correct sign of the
perturbation. Synthetic data only, no network.

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the script is loaded from its path.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name):
    """Import scripts/<name>.py as a module and return it."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script("convert_seismic_models")
fixes = _load_script("fix_seismic_datasets")


def test_voigt_vs_values():
    """Equal vsh and vsv give that speed; the weights are 2/3 for vsv and 1/3 for vsh."""
    np.testing.assert_allclose(convert.voigt_vs(np.array([4500.0]), np.array([4500.0])), [4500.0])
    # vsv = 0 leaves vsh / sqrt(3); vsh = 0 leaves vsv sqrt(2/3).
    np.testing.assert_allclose(convert.voigt_vs(np.array([0.0]), np.array([3.0])), [np.sqrt(3.0)])
    np.testing.assert_allclose(convert.voigt_vs(np.array([3.0]), np.array([0.0])), [3.0 * np.sqrt(2.0 / 3.0)])


def _fields(rng, n=200):
    """Synthetic vsh, vsv and a vs that is their Voigt average, with NaN layers in vs."""
    vsv = 4400.0 + 50.0 * rng.standard_normal(n)
    vsh = 4550.0 + 50.0 * rng.standard_normal(n)
    vs = convert.voigt_vs(vsv, vsh)
    vs_gappy = vs.copy()
    vs_gappy[50:150] = np.nan  # the isotropic component is missing here
    return vs, vs_gappy, vsh, vsv


def test_fill_where_vsh_and_vsv_exist():
    """NaN points of vs are filled with the Voigt average; finite vs does not change."""
    rng = np.random.default_rng(0)
    vs, vs_gappy, vsh, vsv = _fields(rng)
    vsh[60:70] = np.nan  # no anisotropic data here: vs stays NaN
    filled, n_filled, max_rel = convert.fill_vs_from_voigt(vs_gappy, vsh, vsv)
    assert n_filled == 90
    assert max_rel < 1e-12
    assert np.all(np.isnan(filled[60:70]))
    np.testing.assert_array_equal(filled[:50], vs_gappy[:50])
    np.testing.assert_array_equal(filled[150:], vs_gappy[150:])
    np.testing.assert_allclose(filled[70:150], vs[70:150], rtol=1e-14)
    # The NaN pattern of the result is the NaN pattern of vsh | vsv.
    np.testing.assert_array_equal(np.isnan(filled), np.isnan(vsh) | np.isnan(vsv))


def test_no_fill_when_vs_is_not_the_voigt_average():
    """If the existing vs is another average, nothing is filled and a warning is issued."""
    rng = np.random.default_rng(1)
    _, vs_gappy, vsh, vsv = _fields(rng)
    # Replace the existing vs by the swapped formula sqrt((2 vsh^2 + vsv^2) / 3).
    swapped = np.sqrt((2.0 * vsh**2 + vsv**2) / 3.0)
    vs_other = np.where(np.isnan(vs_gappy), np.nan, swapped)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filled, n_filled, max_rel = convert.fill_vs_from_voigt(vs_other, vsh, vsv)
    assert n_filled == 0
    assert max_rel > convert.VOIGT_MATCH_TOLERANCE
    assert any("not filled" in str(w.message) for w in caught)
    np.testing.assert_array_equal(filled, vs_other)


def test_fix_mitp08_vp_sign():
    """vp built with (1 - dvp/100) becomes (1 + dvp/100) times the same reference."""
    rng = np.random.default_rng(2)
    reference = np.full(100, 10.2)  # ak135 at one depth, km/s
    dvp = 0.5 * rng.standard_normal(100)  # percent
    vp_wrong = (1.0 - dvp / 100.0) * reference
    vp_fixed = convert.fix_mitp08_vp(vp_wrong, dvp)
    np.testing.assert_allclose(vp_fixed, (1.0 + dvp / 100.0) * reference, rtol=1e-14)
    assert np.corrcoef(vp_fixed - vp_fixed.mean(), dvp)[0, 1] > 0.999


def _point_cloud(depths_km, n=300):
    """Synthetic layered point cloud: n points per layer on a Fibonacci sphere.

    Returns the Cartesian coordinates in m (layers in the order of
    `depths_km`), the latitude of each point in degrees, and the depth of
    each point in km.
    """
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * i / n))
    lon = np.degrees(np.pi * (1.0 + 5.0**0.5) * i) % 360.0
    unit = np.column_stack((np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                            np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                            np.sin(np.radians(lat))))
    coords = np.vstack([unit * (fixes.R_EARTH - d * 1e3) for d in depths_km])
    return coords, np.tile(lat, len(depths_km)), np.repeat(np.asarray(depths_km, float), n)


def test_seismic_fix_functions():
    """The array functions of fix_seismic_datasets.py on a three-layer point cloud.

    - layer_index groups the points by radius.
    - kms_to_ms converts a km/s field once and leaves an m/s field alone.
    - fill_empty_layers copies the source only into layers where the target is all NaN.
    - block_values gives each point the valid cell of its source row nearest in longitude.
    - restore_voigt_vs repairs vs where it is not the Voigt average of vsh and vsv.
    - depth_range_km gives the depths where a field has values.
    - drop_surface_layer removes the 0 km layer; clip_roundoff_negatives zeros only round-off.
    - append_sentence ends the comment with a full stop and keeps its leading newline.
    """
    coords, lat, depth = _point_cloud([0.0, 100.0, 1000.0])
    radii, index = fixes.layer_index(coords)
    assert len(radii) == 3 and np.all(np.bincount(index) == 300)

    vs_kms = np.full(900, 4.5)
    vs_ms, changed = fixes.kms_to_ms(vs_kms)
    assert changed and np.allclose(vs_ms, 4500.0)
    # A field already in m/s is not converted again
    _, changed_again = fixes.kms_to_ms(vs_ms)
    assert not changed_again

    # Anisotropic field missing at 1000 km, isotropic field complete
    vp = 8000.0 + depth
    vph = np.where(depth < 500.0, vp + 50.0, np.nan)
    filled, n = fixes.fill_empty_layers(vph, vp, index)
    assert n == 1
    np.testing.assert_array_equal(filled[depth == 1000.0], vp[depth == 1000.0])
    np.testing.assert_array_equal(filled[depth < 500.0], vph[depth < 500.0])

    # Equal-area blocks on a 5-degree grid: the polar row has two blocks,
    # at longitudes -90 and 90; points take the value of the nearer one
    grid_lat = np.array([82.5, 87.5])
    grid_lon = np.arange(-177.5, 180.0, 5.0)
    grid = np.full((2, 72), np.nan)
    grid[0, :] = 1.0
    grid[1, np.argmin(abs(grid_lon + 92.5))] = 2.0
    grid[1, np.argmin(abs(grid_lon - 87.5))] = 3.0
    values = fixes.block_values(np.array([88.0, 89.0, 89.0, 83.0]), np.array([-60.0, 100.0, 175.0, 0.0]),
                                grid_lat, grid_lon, grid)
    np.testing.assert_array_equal(values, [2.0, 3.0, 3.0, 1.0])

    # vs with zeros mixed in, and a value where vsh and vsv are missing
    vsv = np.array([4500.0, 4500.0, np.nan, 4600.0])
    vsh = np.array([4600.0, 4600.0, np.nan, 4700.0])
    vs = fixes.voigt_vs(vsv, vsh)
    vs[1] *= 0.8
    vs[2] = 0.0
    repaired, n_nan, n_avg = fixes.restore_voigt_vs(vs, vsh, vsv)
    assert (n_nan, n_avg) == (1, 1)
    np.testing.assert_allclose(repaired[[0, 1, 3]], fixes.voigt_vs(vsv, vsh)[[0, 1, 3]])
    assert np.isnan(repaired[2])

    np.testing.assert_allclose(fixes.depth_range_km(vph, radii, index), [0.0, 100.0], atol=1e-6)

    kept, fields, n = fixes.drop_surface_layer(coords, {"vs": vs_ms})
    assert n == 300 and len(kept) == 600 and len(fields["vs"]) == 600

    clipped, n = fixes.clip_roundoff_negatives(np.array([-2e-13, -5.0, 0.0, 3.0]))
    assert n == 1
    np.testing.assert_array_equal(clipped, [0.0, -5.0, 0.0, 3.0])

    assert fixes.append_sentence("\nRemapped (x@y.edu)", "In percent.") == "\nRemapped (x@y.edu). In percent."
    assert fixes.append_sentence("", "In percent.") == "In percent."


def test_fix_dataset_per_model():
    """fix_dataset applies the right fixes to each model and the result passes check_dataset.

    Synthetic stand-ins for LLNL-G3D-JPS (surface level, fraction perturbations,
    round-off negatives in water), PMEAN (empty vph layer), SEMUM2 (vs ending
    above vsv), SL2013NA (regional) and HMSL-S06 (km/s, percent).
    """
    coords, lat, depth = _point_cloud([0.0, 3.2, 500.0, 1000.0])

    # LLNL-G3D-JPS: negative surface level, water level with round-off, fractions
    vs = np.where(depth == 0.0, -2875.7, np.where(depth == 3.2, -2e-13, 4500.0 + depth))
    dvs = 0.01 * np.sin(np.radians(lat))
    jps_fields = {"vs": vs, "dvs": dvs, "vp": vs + 3000.0, "dvp": dvs}
    c, f, a, _ = fixes.fix_dataset("LLNL-G3D-JPS", coords, jps_fields, {"comment": "x"})
    assert len(c) == 3 * 300 and np.nanmin(f["vs"]) == 0.0
    np.testing.assert_allclose(f["dvs"], 100.0 * dvs[depth > 0.0])
    assert a["perturbation_units"] == "percent" and fixes.check_dataset("LLNL-G3D-JPS", c, f, a) == []

    # PMEAN: vph missing below 500 km is set to vp
    vp = 8000.0 + depth
    vph = np.where(depth <= 500.0, vp + 20.0, np.nan)
    c, f, a, _ = fixes.fix_dataset("PMEAN", coords, {"vp": vp, "vph": vph, "vpv": vph.copy()}, {})
    np.testing.assert_array_equal(f["vph"][depth == 1000.0], vp[depth == 1000.0])
    assert fixes.check_dataset("PMEAN", c, f, a) == []

    # SEMUM2: vs and vsh end at 500 km, vsv goes on; NaN stays, ranges are
    # recorded; a zero in vs where vsh and vsv are missing becomes NaN
    vsv = 4500.0 + depth
    vsh = np.where(depth <= 500.0, vsv + 30.0, np.nan)
    vs = fixes.voigt_vs(vsv, vsh)
    vsv[:5] = np.nan
    vsh[:5] = np.nan
    vs[:5] = 0.0
    c, f, a, _ = fixes.fix_dataset("SEMUM2", coords, {"vs": vs, "vsh": vsh, "vsv": vsv}, {})
    assert np.isnan(f["vs"][depth == 1000.0]).all() and np.isnan(f["vs"][:5]).all()
    np.testing.assert_allclose(a["depth_range_km_vs"], [0.0, 500.0], atol=1e-6)
    np.testing.assert_allclose(a["depth_range_km_vsv"], [0.0, 1000.0], atol=1e-6)
    assert fixes.check_dataset("SEMUM2", c, f, a) == []

    # SL2013NA: flagged as regional
    _, _, a, _ = fixes.fix_dataset("SL2013NA", coords, {"vsv": 4500.0 + depth}, {})
    assert a["regional"] == "true"

    # HMSL-S06: vs in km/s becomes m/s, dvs gets the percent unit, the warning is added
    c, f, a, _ = fixes.fix_dataset("HMSL-S06", coords, {"dvs": dvs * 100.0, "vs": 4.5 + 0.0 * depth}, {})
    np.testing.assert_allclose(f["vs"], 4500.0)
    assert a["perturbation_units"] == "percent" and "ak135" in a["comment"]
    assert fixes.check_dataset("HMSL-S06", c, f, a) == []
