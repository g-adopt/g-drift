import numpy as np
import pytest
from gdrift.anelasticity import (
    BaseAnelasticityModel,
    CammaranoAnelasticityModel,
    GoesAnelasticityModel,
    apply_anelastic_correction,
)
from gdrift import ThermodynamicModel


class MockSolidus:
    """Mock solidus with constant temperature for analytic formula tests."""

    def __init__(self, T_solidus=2000.0):
        self.T_solidus = T_solidus

    def at_depth(self, depths):
        return np.full_like(np.asarray(depths, dtype=float), self.T_solidus)

    def min_max_depth(self):
        return (0.0, 3000e3)


# ---------------------------------------------------------------------------
# Cammarano formula tests
# ---------------------------------------------------------------------------

def test_cammarano_q_shear_formula():
    """Hand-computed Q value with constant parameters.

    With B=1.0, g=20, a=0.2, omega=1.0, T_solidus=2000, T=1000:
        Q = 1.0 * 1.0^0.2 * exp(0.2 * 20 * 2000 / 1000)
          = exp(8) ~ 2980.96
    """
    solidus = MockSolidus(T_solidus=2000.0)
    model = CammaranoAnelasticityModel(
        B=lambda x: 1.0,
        g=lambda x: 20.0,
        a=lambda x: 0.2,
        solidus=solidus,
        omega=lambda x: 1.0,
    )
    Q = model.compute_Q_shear(np.array([500e3]), np.array([1000.0]))
    expected = np.exp(0.2 * 20.0 * 2000.0 / 1000.0)
    np.testing.assert_allclose(Q, expected, rtol=1e-10)


def test_cammarano_q_shear_temperature_dependence():
    """Higher temperature should give lower Q (more attenuation at lower T)."""
    solidus = MockSolidus(T_solidus=2000.0)
    model = CammaranoAnelasticityModel(
        B=lambda x: 1.0,
        g=lambda x: 20.0,
        a=lambda x: 0.2,
        solidus=solidus,
        omega=lambda x: 1.0,
    )
    Q_cold = model.compute_Q_shear(np.array([500e3]), np.array([1000.0]))
    Q_hot = model.compute_Q_shear(np.array([500e3]), np.array([3000.0]))
    # Colder temperatures produce higher Q (less attenuation)
    assert Q_cold > Q_hot


def test_cammarano_q_bulk():
    """Bulk Q should switch at 660 km (1000 above, 10000 below)."""
    solidus = MockSolidus()
    model = CammaranoAnelasticityModel(
        B=lambda x: 1.0,
        g=lambda x: 20.0,
        a=lambda x: 0.2,
        solidus=solidus,
        Q_bulk=lambda x: np.where(x < 660e3, 1e3, 1e4),
        omega=lambda x: 1.0,
    )
    depths = np.array([300e3, 1000e3])
    Q_bulk = model.compute_Q_bulk(depths, np.array([1500.0, 1500.0]))
    np.testing.assert_allclose(Q_bulk, [1e3, 1e4])


# ---------------------------------------------------------------------------
# Goes formula tests
# ---------------------------------------------------------------------------

def test_goes_q_shear_formula():
    """Hand-computed Q for Goes activation energy model.

    With A=1.0, H*=500e3 J/mol, V*=0 (ignore pressure term), a=0.15,
    omega=1.0, T=1600 K, P=0:
        Q = 1.0 * 1.0^0.15 * exp(0.15 * 500e3 / (8.314 * 1600))
          = exp(0.15 * 500000 / 13302.4)
          = exp(5.638...)
    """
    model = GoesAnelasticityModel(A=1.0, H_star=500e3, V_star=0.0, a=0.15, omega=1.0)
    # At zero depth, PREM pressure is ~0, so the P*V* term vanishes
    Q = model.compute_Q_shear(np.array([0.0]), np.array([1600.0]))
    R = 8.314
    expected = np.exp(0.15 * 500e3 / (R * 1600.0))
    np.testing.assert_allclose(Q, expected, rtol=1e-3)


def test_goes_q_shear_temperature_dependence():
    """Higher temperature should give lower Q (less attenuation at higher T)."""
    model = GoesAnelasticityModel(A=0.148, H_star=500e3, V_star=20e-6, a=0.15)
    Q_cold = model.compute_Q_shear(np.array([100e3]), np.array([1000.0]))
    Q_hot = model.compute_Q_shear(np.array([100e3]), np.array([2000.0]))
    assert Q_cold > Q_hot


def test_goes_q_bulk():
    """Goes bulk Q is a constant (Q_K = 1000 by default)."""
    model = GoesAnelasticityModel(A=0.148, H_star=500e3, V_star=20e-6, a=0.15)
    Q_bulk = model.compute_Q_bulk(np.array([300e3, 1000e3]), np.array([1500.0, 1500.0]))
    assert Q_bulk == 1000.0


def test_goes_deep_depth_high_q():
    """Depths below 660 km should produce very high Q (no attenuation)."""
    model = GoesAnelasticityModel(A=0.148, H_star=500e3, V_star=20e-6, a=0.15)
    Q = model.compute_Q_shear(np.array([100e3, 800e3]), np.array([1600.0, 1600.0]))
    assert Q[0] < 1e9  # upper mantle: normal Q
    assert Q[1] == 1e10  # below 660 km: clamped


def test_goes_deep_depth_warning():
    """A warning should be issued when querying depths beyond max_depth."""
    model = GoesAnelasticityModel(A=0.148, H_star=500e3, V_star=20e-6, a=0.15)
    with pytest.warns(UserWarning, match="calibrated for the upper mantle"):
        model.compute_Q_shear(np.array([800e3]), np.array([1600.0]))


# ---------------------------------------------------------------------------
# Factory method tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_profile", ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
def test_cammarano_from_q_profile_valid(q_profile):
    """All six Cammarano Q-profiles should produce valid models."""
    model = CammaranoAnelasticityModel.from_q_profile(q_profile)
    assert isinstance(model, CammaranoAnelasticityModel)
    # Should produce positive Q values
    Q = model.compute_Q_shear(np.array([500e3]), np.array([2000.0]))
    assert Q > 0


def test_cammarano_from_q_profile_invalid():
    """Invalid Q-profile should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown Q-profile"):
        CammaranoAnelasticityModel.from_q_profile("Q99")


@pytest.mark.parametrize("q_profile", ["Q1", "Q2"])
def test_goes_from_q_profile_valid(q_profile):
    """Both Goes Q-profiles should produce valid models."""
    model = GoesAnelasticityModel.from_q_profile(q_profile)
    assert isinstance(model, GoesAnelasticityModel)
    Q = model.compute_Q_shear(np.array([100e3]), np.array([2000.0]))
    assert Q > 0


def test_goes_from_q_profile_invalid():
    """Invalid Q-profile should raise ValueError for Goes model."""
    with pytest.raises(ValueError, match="Unknown Q-profile"):
        GoesAnelasticityModel.from_q_profile("Q4")


# ---------------------------------------------------------------------------
# Depth switching test
# ---------------------------------------------------------------------------

def test_cammarano_depth_switching():
    """Q at 300 km should differ from Q at 1000 km due to 660 km boundary."""
    model = CammaranoAnelasticityModel.from_q_profile("Q3")
    T = np.array([2000.0])
    Q_shallow = model.compute_Q_shear(np.array([300e3]), T)
    Q_deep = model.compute_Q_shear(np.array([1000e3]), T)
    assert not np.isclose(Q_shallow, Q_deep)


# ---------------------------------------------------------------------------
# Solidus factory test
# ---------------------------------------------------------------------------

def test_build_ghelichkhan_solidus():
    """Factory should return a SplineProfile with valid solidus values."""
    from gdrift.profile import SplineProfile
    solidus = BaseAnelasticityModel.build_ghelichkhan_solidus()
    assert isinstance(solidus, SplineProfile)
    # Solidus temperatures should be positive across the mantle
    test_depths = np.array([100e3, 500e3, 1000e3, 2000e3, 2800e3])
    temps = solidus.at_depth(test_depths)
    assert np.all(temps > 0)
    # Should span the full mantle
    d_min, d_max = solidus.min_max_depth()
    assert d_min < 10e3
    assert d_max >= 2900e3


# ---------------------------------------------------------------------------
# Integration tests (with real ThermodynamicModel)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slb21_model():
    """Load SLB_21 pyroliteCFMAS once for integration tests."""
    return ThermodynamicModel(
        model="SLB_21",
        composition="pyroliteCFMAS",
        temps=np.linspace(300, 4000, 10),
        depths=np.linspace(100e3, 2800e3, 10),
    )


def test_apply_anelastic_correction_reduces_vs(slb21_model):
    """Anelastic Vs should be less than elastic Vs."""
    anelastic_model = CammaranoAnelasticityModel.from_q_profile("Q3")
    corrected = apply_anelastic_correction(slb21_model, anelastic_model)

    elastic_vs = slb21_model.compute_swave_speed().get_vals()
    corrected_vs = corrected.compute_swave_speed().get_vals()

    # All corrected values should be less than elastic (positive Q -> reduction)
    assert np.all(corrected_vs < elastic_vs)


def test_apply_anelastic_correction_reduces_vp(slb21_model):
    """Anelastic Vp should be less than elastic Vp."""
    anelastic_model = CammaranoAnelasticityModel.from_q_profile("Q3")
    corrected = apply_anelastic_correction(slb21_model, anelastic_model)

    elastic_vp = slb21_model.compute_pwave_speed().get_vals()
    corrected_vp = corrected.compute_pwave_speed().get_vals()

    assert np.all(corrected_vp < elastic_vp)


def test_apply_anelastic_correction_goes(slb21_model):
    """Goes model should also reduce velocities.

    At very low temperatures or depths > 660 km the correction is negligible
    (Q is enormous), so we check <= overall and < for the upper-mantle, warm part.
    """
    anelastic_model = GoesAnelasticityModel.from_q_profile("Q1")
    corrected = apply_anelastic_correction(slb21_model, anelastic_model)

    elastic_vs = slb21_model.compute_swave_speed().get_vals()
    corrected_vs = corrected.compute_swave_speed().get_vals()

    assert np.all(corrected_vs <= elastic_vs)
    # In the warm upper mantle, the correction should be strictly reducing
    assert np.any(corrected_vs < elastic_vs)
