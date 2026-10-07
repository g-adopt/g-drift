"""Tests for the regularisation of thermodynamic tables.

`regularise_thermodynamic_table` differentiates each depth row of a table
along temperature, replaces derivatives outside `regular_range`, integrates
back, and anchors each row to the raw value at a reference temperature. Two
properties follow from that construction and are tested here:

1. Where no derivative is replaced, the regularised table equals the raw
   table. Differencing and integrating must be exact inverses, so the table
   cannot move along the temperature axis.
2. At the anchor temperature of each depth, the regularised value equals the
   raw value, also when the anchor temperature lies between two table nodes.
"""

import numpy as np

import gdrift
from gdrift.mineralogy import Table, default_regular_range, derive_then_integrate


def _regularise_table(table, profile, regular_range=default_regular_range):
    """Regularise a single Table in the same way as `regularise_thermodynamic_table`.

    `derive_then_integrate` returns each row minus its value at the anchor
    temperature. Adding the raw value at the anchor temperature (linear
    interpolation along T, as in ThermodynamicModel) gives the regularised
    table. This lets the tests use synthetic tables without a dataset.
    """
    temperatures = table.get_y()
    anchors = profile.at_depth(table.get_x())
    raw_at_anchor = np.array([np.interp(t_a, temperatures, row) for t_a, row in zip(anchors, table.get_vals())])
    return derive_then_integrate(table, profile, regular_range) + raw_at_anchor[:, None]


def test_round_trip_and_anchor_on_smooth_table():
    """A table without out-of-range derivatives comes back unchanged.

    The table decreases with T everywhere (dV/dT = -0.3 - 8e-5 T - ... < 0),
    so no derivative is replaced. The temperature grid has two different
    spacings to cover a non-uniform grid, and the anchor temperatures lie
    between nodes. A half-node shift of the table along T would show up here
    as errors of several m/s.
    """
    depths = np.array([0.0, 1.0e6, 2.0e6])
    temperatures = np.concatenate([np.linspace(300.0, 1000.0, 8), np.linspace(1050.0, 4000.0, 30)])
    D, T = np.meshgrid(depths, temperatures, indexing="ij")
    # Smooth, curved in T, depth dependent, strictly decreasing in T
    values = 7000.0 - 0.3 * T - 4.0e-5 * T**2 + 1.0e-3 * D - 1.0e-10 * D * T
    table = Table(depths, temperatures, values, name="v_s")

    # Anchor temperatures that are not table nodes
    profile = gdrift.SplineProfile(depth=depths, value=np.array([1234.5, 1876.3, 2611.1]))

    regularised = _regularise_table(table, profile)

    # The raw table is returned to round-off
    np.testing.assert_allclose(regularised, values, rtol=1e-12, atol=1e-9)

    # The rows from derive_then_integrate are zero at the anchor temperature
    anchored = derive_then_integrate(table, profile, default_regular_range)
    at_anchor = [np.interp(t_a, temperatures, row) for t_a, row in zip(profile.at_depth(depths), anchored)]
    np.testing.assert_allclose(at_anchor, 0.0, atol=1e-9)


def test_single_phase_step_is_removed():
    """One cell with a positive slope takes the slope of its neighbours.

    Every row is V = 7000 - 0.5 T (m/s). In the middle row the values above
    1200 K are raised by 40 m/s, which gives one cell with a slope of
    -0.5 + 40/50 = +0.3 m/s/K, as a phase transition would. That cell is
    outside the default range (dV/dT < 0) and its slope is replaced by the
    inverse-distance mean of the nearest accepted slopes, all of which are
    -0.5. The anchor is at 1612.5 K, above the step, so the regularised middle
    row is the line through the raised part: 7040 - 0.5 T at every node.
    The other rows have nothing to replace and stay unchanged.
    """
    depths = np.array([500.0e3, 600.0e3, 700.0e3])
    temperatures = np.arange(300.0, 2001.0, 50.0)
    values = np.tile(7000.0 - 0.5 * temperatures, (len(depths), 1))
    values[1, temperatures >= 1200.0] += 40.0
    table = Table(depths, temperatures, values, name="v_s")
    profile = gdrift.SplineProfile(depth=depths, value=np.full(len(depths), 1612.5))

    regularised = _regularise_table(table, profile)

    # Only the cell with the jump had a positive slope
    assert np.count_nonzero(np.diff(values, axis=1) >= 0) == 1

    # The step is gone: the middle row is a straight line through the raised part
    np.testing.assert_allclose(regularised[1], 7040.0 - 0.5 * temperatures, rtol=1e-12)

    # Rows without a replaced slope are unchanged
    np.testing.assert_allclose(regularised[[0, 2]], values[[0, 2]], rtol=1e-12)


def test_regularised_slb_table_matches_raw_table():
    """The public function on a real table: raw rows kept, anchor exact, no positive slopes.

    For density, S-wave and P-wave speed of SLB_21 pyrolite:
    - in every depth row where the raw table already decreases with T, the
      regularised row equals the raw row;
    - at the anchor temperature of every table depth, the regularised model
      gives the raw value through the public `temperature_to_*` methods;
    - the regularised table decreases with T everywhere.
    """
    slb = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    depths = slb.get_depths()

    # A mantle-like anchor profile, offset so that it does not hit table nodes
    profile = gdrift.SplineProfile(
        depth=np.array([0.0, 100e3, 660e3, 2700e3, 2900e3]),
        value=np.array([413.7, 1613.7, 1893.7, 2533.7, 3313.7]),
        extrapolate=True,
    )
    regularised = gdrift.regularise_thermodynamic_table(slb, profile)
    anchors = profile.at_depth(depths)

    for raw_table, reg_table, raw_lookup, reg_lookup in (
        (slb._tables["rho"], regularised._tables["rho"], slb.temperature_to_rho, regularised.temperature_to_rho),
        (slb.compute_swave_speed(), regularised.compute_swave_speed(), slb.temperature_to_vs, regularised.temperature_to_vs),
        (slb.compute_pwave_speed(), regularised.compute_pwave_speed(), slb.temperature_to_vp, regularised.temperature_to_vp),
    ):
        raw = np.asarray(raw_table.get_vals())
        reg = reg_table.get_vals()

        # Rows where nothing is replaced (every raw slope is negative)
        untouched = np.all(np.diff(raw, axis=1) < 0, axis=1)
        assert untouched.sum() > len(depths) // 4, raw_table.get_name()
        np.testing.assert_allclose(reg[untouched], raw[untouched], rtol=1e-12, err_msg=raw_table.get_name())

        # Anchor condition at every table depth, through the public interpolation
        np.testing.assert_allclose(reg_lookup(anchors, depths), raw_lookup(anchors, depths), rtol=1e-12, err_msg=raw_table.get_name())

        # No temperature cell of the regularised table has a positive slope.
        # The bound is round-off, not 0: some raw rows are constant at high T
        # (for example at 132 km above 4150 K), with slopes of order -1e-13
        # that are accepted and stay at 0 after the integration.
        assert np.diff(reg, axis=1).max() < 1e-9, raw_table.get_name()
