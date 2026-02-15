"""Unit tests for gdrift.adiabat module."""

import numpy as np
import pytest

import gdrift
from gdrift.profile import SplineProfile


class TestPremGravityProfile:
    """Tests for prem_gravity_profile()."""

    def test_returns_spline_profile(self):
        profile = gdrift.prem_gravity_profile()
        assert isinstance(profile, SplineProfile)

    def test_surface_gravity(self):
        profile = gdrift.prem_gravity_profile()
        g_surface = profile.at_depth(0)
        np.testing.assert_allclose(g_surface, 9.8, atol=0.2)

    def test_cmb_gravity(self):
        profile = gdrift.prem_gravity_profile()
        g_cmb = profile.at_depth(2890e3)
        np.testing.assert_allclose(g_cmb, 10.7, atol=0.5)


class TestComputeAdiabat:
    """Tests for compute_adiabat()."""

    @pytest.fixture(scope="class")
    def adiabat(self):
        tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        return gdrift.compute_adiabat(tm, T0=1600)

    def test_returns_dict_with_expected_keys(self, adiabat):
        expected_keys = {
            "depths", "temperature", "rho", "alpha", "Cp", "V",
            "Cv", "beta", "gamma", "Cp_SI", "Cv_SI", "gravity", "Di",
        }
        assert expected_keys.issubset(adiabat.keys())

    def test_surface_temperature_equals_T0(self, adiabat):
        np.testing.assert_allclose(adiabat["temperature"][0], 1600.0, rtol=1e-10)

    def test_temperature_increases_monotonically(self, adiabat):
        assert np.all(np.diff(adiabat["temperature"]) >= 0)

    def test_di_positive_and_reasonable(self, adiabat):
        # NOTE: Surface-based Di (~1.6) is higher than the commonly cited
        # depth-averaged value (~0.5-0.7). See CLAUDE.md for investigation note.
        assert 0.3 < adiabat["Di"] < 3.0

    def test_custom_depths(self):
        tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        depths = np.linspace(0, 1000e3, 50)
        result = gdrift.compute_adiabat(tm, T0=1600, depths=depths)
        assert len(result["depths"]) == 50
        np.testing.assert_array_equal(result["depths"], depths)

    def test_custom_property_names(self):
        tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        result = gdrift.compute_adiabat(
            tm, T0=1600, property_names=["rho", "alpha"])
        assert "rho" in result
        assert "alpha" in result
        assert "beta" not in result

    def test_higher_T0_gives_higher_temperature(self):
        tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        depths = np.linspace(0, 1000e3, 50)
        result_1600 = gdrift.compute_adiabat(tm, T0=1600, depths=depths)
        result_1700 = gdrift.compute_adiabat(tm, T0=1700, depths=depths)
        assert np.all(result_1700["temperature"] > result_1600["temperature"])
