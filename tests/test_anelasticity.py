import numpy as np
from gdrift.anelasticity import CammaranoAnelasticityModel, apply_anelastic_correction
from gdrift import ThermodynamicModel

# Mock solidus function


class MockSolidus:
    def __init__(self, depths, solidus_values):
        self.depths = np.array(depths)
        self.solidus_values = np.array(solidus_values)

    def at_depth(self, depths):
        return np.interp(depths, self.depths, self.solidus_values)

# Test CammaranoAnelasticityModel


def test_cammarano_anelasticity_model():
    # Define mock solidus
    mock_solidus = MockSolidus(
        depths=[0, 660e3, 2890e3],
        solidus_values=[1200, 1500, 1800]
    )

    # Build anelasticity model
    model = CammaranoAnelasticityModel(
        B=lambda x: np.where(x < 660e3, 1.1, 20),
        g=lambda x: np.where(x < 660e3, 20, 10),
        a=lambda x: 0.2,
        solidus=mock_solidus,
        Q_bulk=lambda x: np.where(x < 660e3, 1e3, 1e4),
        omega=lambda x: 1.0
    )

    # Test Q_shear computation
    depths = np.array([100e3, 500e3, 1000e3])
    temperatures = np.array([1300, 1400, 1700])
    Q_shear = model.compute_Q_shear(depths, temperatures)

    # Expected values based on mock parameters
    expected_Q_shear = model.B(depths) * (model.omega(depths) ** model.a(depths)) * np.exp(
        (model.a(depths) * model.g(depths) * mock_solidus.at_depth(depths)) / temperatures
    )

    assert np.allclose(
        Q_shear, expected_Q_shear), "Q_shear calculation failed."

    # Test Q_bulk computation
    Q_bulk = model.compute_Q_bulk(depths, temperatures)
    expected_Q_bulk = np.where(depths < 660e3, 1e3, 1e4)
    assert np.allclose(Q_bulk, expected_Q_bulk), "Q_bulk calculation failed."

# Test Anelastic Correction Application


def test_apply_anelastic_correction():
    # Build mock solidus and anelasticity model
    mock_solidus = MockSolidus(
        depths=[0, 660e3, 2890e3],
        solidus_values=[1200, 1500, 1800]
    )
    anelasticity_model = CammaranoAnelasticityModel(
        B=lambda x: np.where(x < 660e3, 1.1, 20),
        g=lambda x: np.where(x < 660e3, 20, 10),
        a=lambda x: 0.2,
        solidus=mock_solidus,
        Q_bulk=lambda x: np.where(x < 660e3, 1e3, 1e4),
        omega=lambda x: 1.0
    )

    # Load thermodynamic model
    thermodynamic_model = ThermodynamicModel(
        model="SLB_24",
        composition="pyroliteCFMAS",
        temps=np.linspace(300, 4000, 10),
        depths=np.linspace(0, 2890e3, 10)
    )

    # Apply anelastic correction
    corrected_model = apply_anelastic_correction(
        thermodynamic_model, anelasticity_model)

    # Check that corrected S-wave speeds differ from elastic ones
    elastic_speeds = thermodynamic_model.compute_swave_speed().get_vals()
    corrected_speeds = corrected_model.compute_swave_speed().get_vals()

    assert not np.allclose(
        elastic_speeds, corrected_speeds), "Anelastic correction not applied correctly."

    # Check that corrected P-wave speeds differ from elastic ones
    elastic_p_speeds = thermodynamic_model.compute_pwave_speed().get_vals()
    corrected_p_speeds = corrected_model.compute_pwave_speed().get_vals()

    assert not np.allclose(
        elastic_p_speeds, corrected_p_speeds), "Anelastic correction not applied correctly for P-waves."
