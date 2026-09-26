"""Tests for the boundary-order fix of the LLNL-G3D-JPS conversion.

The LLNL source stores each boundary as two levels about 0.1 km apart. At five
boundaries (upper/middle crust, middle/lower crust, Moho, 410 km, 660 km) it
puts the level with the lower unit's values at the smaller depth.
``scripts/convert_llnl_jps.py:fix_boundary_order`` swaps the two depths of
each of these pairs. These tests use synthetic levels, so they need no network
and no source file.

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the script is loaded from its path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_llnl_jps.py"


def _load_script():
    """Import scripts/convert_llnl_jps.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("convert_llnl_jps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _load_script()

# Latitudes of the synthetic grid. Two latitudes and three longitudes are
# enough, because every level holds one constant value.
LAT = np.array([-30.0, 30.0])
NUM_LON = 3


def _levels(reversed_pairs=True):
    """Return synthetic (depth_km, r_m, vp) arrays like the LLNL source.

    The levels are a chained triple of close levels above the crust pairs
    (like the 5.4-5.6 km cluster of the source), the five boundary pairs and
    one mantle level at 72.4574 km. Each pair has an upper-unit value and a
    lower-unit value; the lower unit is faster. With reversed_pairs=True the
    faster value sits at the smaller depth of each pair, as in the source.
    The depths carry extra digits, as the stored depths of the source do.
    """
    # (depth km, Vp km/s) of the triple: close levels whose order is not
    # touched by the fix.
    triple = [(5.38521, 3.810), (5.48521, 1.500), (5.61171, 2.216)]
    # Upper-unit and lower-unit Vp at each of the five boundaries.
    pair_vp = [(5.408, 6.477), (6.477, 7.044), (7.044, 8.078), (8.907, 9.140), (10.251, 10.732)]
    depth, vp = [d for d, _ in triple], [v for _, v in triple]
    for (shallow, deep, _), (vp_upper, vp_lower) in zip(convert.REVERSED_BOUNDARY_PAIRS_KM, pair_vp):
        # Add a few metres below the printed four decimals, well inside the
        # 1 m tolerance.
        depth += [shallow + 4e-6, deep + 4e-6]
        vp += [vp_lower, vp_upper] if reversed_pairs else [vp_upper, vp_lower]
    depth.append(72.4574)
    vp.append(8.085)
    depth = np.array(depth)
    # Radius of each level in m, as in the source: r = 6371 km - depth.
    r = 6371e3 - depth * 1e3
    # Constant value over the (lat, lon) plane of each level.
    vp_grid = np.array(vp)[:, None, None] * np.ones((1, LAT.size, NUM_LON))
    return depth, r, vp_grid


def test_mean_vp_increases_with_depth_after_fix():
    """After the fix, Vp is non-decreasing with depth from the first pair down."""
    depth, r, vp = _levels()
    depth_fixed, r_fixed = convert.fix_boundary_order(depth, r, vp, LAT)
    order = np.argsort(depth_fixed)
    vp_sorted = vp[order, 0, 0]
    depth_sorted = depth_fixed[order]
    below = depth_sorted >= convert.REVERSED_BOUNDARY_PAIRS_KM[0][0] - 1e-3
    assert np.all(np.diff(vp_sorted[below]) >= 0)
    # The radii stay consistent with the depths.
    np.testing.assert_allclose(r_fixed, 6371e3 - depth_fixed * 1e3, rtol=0, atol=1e-6)


def test_levels_outside_the_pairs_do_not_change():
    """The triple above the pairs and the mantle level keep their depths."""
    depth, r, vp = _levels()
    depth_fixed, _ = convert.fix_boundary_order(depth, r, vp, LAT)
    np.testing.assert_array_equal(depth_fixed[:3], depth[:3])
    assert depth_fixed[-1] == depth[-1]
    # The set of depths does not change: the fix only permutes them.
    np.testing.assert_array_equal(np.sort(depth_fixed), np.sort(depth))


def test_input_arrays_are_not_modified():
    """The function returns copies and leaves its inputs as they are."""
    depth, r, vp = _levels()
    depth_before, r_before = depth.copy(), r.copy()
    convert.fix_boundary_order(depth, r, vp, LAT)
    np.testing.assert_array_equal(depth, depth_before)
    np.testing.assert_array_equal(r, r_before)


def test_source_in_order_raises():
    """A source whose pairs are already in order raises and is not swapped back."""
    depth, r, vp = _levels(reversed_pairs=False)
    with pytest.raises(ValueError, match="not reversed"):
        convert.fix_boundary_order(depth, r, vp, LAT)


def test_second_application_raises():
    """Applying the fix to its own output raises."""
    depth, r, vp = _levels()
    depth_fixed, r_fixed = convert.fix_boundary_order(depth, r, vp, LAT)
    with pytest.raises(ValueError, match="not reversed"):
        convert.fix_boundary_order(depth_fixed, r_fixed, vp, LAT)


def test_missing_pair_raises():
    """A source without one of the pair depths raises."""
    depth, r, vp = _levels()
    # Move the Moho pair away by 10 m, outside the 1 m tolerance.
    moho = np.abs(depth - 26.3947) < 1e-3
    depth = np.where(moho, depth + 0.01, depth)
    with pytest.raises(ValueError, match="expected one level"):
        convert.fix_boundary_order(depth, r, vp, LAT)
