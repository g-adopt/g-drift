# API Reference

Complete API documentation for gdrift. All classes and functions are organized by functionality.

## Core Data Loading

### load_dataset

::: gdrift.io.load_dataset

---

## 1D Radial Profiles

### RadialEarthModel

::: gdrift.profile.RadialEarthModel
    options:
      members:
        - get_profile
        - get_all_profile_names

### RadialEarthModelFromFile

::: gdrift.profile.RadialEarthModelFromFile

### PreliminaryRefEarthModel

::: gdrift.profile.PreliminaryRefEarthModel

### HirschmannSolidus

::: gdrift.profile.HirschmannSolidus

---

## 3D Seismic Models

### SeismicModel

::: gdrift.seismic.SeismicModel
    options:
      members:
        - interpolate
        - set_kernel

The full list of available tomography models is exposed as `gdrift.AVAILABLE_SEISMIC_MODELS`.
See the [Data Catalog](data-catalog.md) for descriptions and citations.

---

## Thermodynamics

### ThermodynamicModel

::: gdrift.mineralogy.ThermodynamicModel
    options:
      members:
        - available_tables
        - temperature_to_property
        - temperature_to_vs
        - temperature_to_vp
        - temperature_to_rho
        - vs_to_temperature
        - vp_to_temperature

### Phase Transition Regularization

::: gdrift.mineralogy.regularise_thermodynamic_table

---

## Anelasticity

### CammaranoAnelasticityModel

::: gdrift.anelasticity.CammaranoAnelasticityModel
    options:
      members:
        - from_q_profile
        - attenuation_coefficient
        - quality_factor

### GoesAnelasticityModel

::: gdrift.anelasticity.GoesAnelasticityModel
    options:
      members:
        - from_q_profile
        - attenuation_coefficient
        - quality_factor

### apply_anelastic_correction

::: gdrift.anelasticity.apply_anelastic_correction

---

## Utility Functions

### Coordinate Transformations

::: gdrift.geodetic_to_cartesian

::: gdrift.cartesian_to_geodetic

::: gdrift.nondimensionalise_coords

::: gdrift.dimensionalise_coords

### Gravity and Pressure

::: gdrift.compute_gravity

::: gdrift.compute_pressure

::: gdrift.compute_mass

---

## Constants

::: gdrift.R_earth

::: gdrift.R_cmb

---

## Usage Examples

See the [Examples](examples/index.md) section for Jupyter notebooks demonstrating these APIs in action.
