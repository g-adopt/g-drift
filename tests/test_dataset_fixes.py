"""Tests for the dataset fixes in scripts/convert_seismic_models.py.

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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_seismic_models.py"


def _load_script():
    """Import scripts/convert_seismic_models.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_seismic_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()


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
