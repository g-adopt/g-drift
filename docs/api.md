# API Reference

Complete API documentation for gdrift. All classes and functions are organized by functionality.

## Core Data Loading

### load_dataset

::: gdrift.io.load_dataset

### DatasetRegistry

::: gdrift.datasetnames.DatasetRegistry
    options:
      members:
        - get_dataset
        - get_datasets_by_type
        - list_all_datasets

---

## 1D Radial Profiles

### SplineProfile

::: gdrift.profile.SplineProfile
    options:
      members:
        - __init__
        - at_depth
        - min_max_depth

### RadialEarthModel

::: gdrift.profile.RadialEarthModel
    options:
      members:
        - __init__
        - get_profile
        - get_all_profile_names

### RadialEarthModelFromFile

::: gdrift.profile.RadialEarthModelFromFile
    options:
      members:
        - __init__

### PreliminaryRefEarthModel

::: gdrift.profile.PreliminaryRefEarthModel
    options:
      members:
        - __init__

### HirschmannSolidus

::: gdrift.profile.HirschmannSolidus
    options:
      members:
        - __init__

---

## 3D Earth Models

### EarthModel3D

::: gdrift.earthmodel3d.EarthModel3D
    options:
      members:
        - __init__
        - interpolate
        - set_kernel
        - get_coordinates

### SeismicModel

::: gdrift.seismic.SeismicModel
    options:
      members:
        - __init__
        - interpolate
        - set_kernel

### Available Seismic Models

::: gdrift.seismic.AVAILABLE_SEISMIC_MODELS

---

## Thermodynamics

### ThermodynamicModel

::: gdrift.mineralogy.ThermodynamicModel
    options:
      members:
        - __init__
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
        - __init__
        - from_q_profile
        - attenuation_coefficient
        - quality_factor

### GoesAnelasticityModel

::: gdrift.anelasticity.GoesAnelasticityModel
    options:
      members:
        - __init__
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

### Mesh Generation

::: gdrift.fibonacci_sphere

---

## Constants

::: gdrift.R_earth

::: gdrift.R_cmb

---

## Dataset Information

### Dataset Types

All datasets in gdrift are categorized by type:

- **Reference Earth Model**: 1D radial profiles (e.g., PREM)
- **Solidus Profile**: Mantle solidus temperature vs. depth
- **Seismic Tomography Model**: 3D seismic velocity anomalies
- **Thermodynamic Model**: Temperature-to-property lookup tables
- **Geodynamic Profile**: Pre-computed adiabatic profiles

### Registered Datasets

#### Reference Earth Models

| Name | Utility Class | Citation |
|------|--------------|----------|
| `1d_prem` | `PreliminaryRefEarthModel` | Dziewonski & Anderson (1981) |

#### Solidus Profiles

| Name | Utility Class | Citation |
|------|--------------|----------|
| `1d_solidus_Andrault_et_al_2011_EPSL` | `RadialEarthModelFromFile` | Andrault et al. (2011) EPSL |
| `1d_solidus_Fiquet_et_al_2010_SCIENCE` | `RadialEarthModelFromFile` | Fiquet et al. (2010) Science |
| `1d_solidus_Ghelichkhan_et_al_2021_GJI` | `RadialEarthModelFromFile` | Ghelichkhan et al. (2021) GJI |
| `1d_solidus_Nomura_et_al_2014_SCIENCE` | `RadialEarthModelFromFile` | Nomura et al. (2014) Science |
| `1d_solidus_Zerr_et_al_1988_SCIENCE` | `RadialEarthModelFromFile` | Zerr & Boehler (1994) Nature |

#### Seismic Tomography Models

See `AVAILABLE_SEISMIC_MODELS` for the complete list of 25 tomography models including:

- S40RTS (Ritsema et al. 2011)
- GLAD-M25 (Lei et al. 2020)
- SEMUCB-WM1 (French & Romanowicz 2014)
- REVEAL (Thrastarson et al. 2024)
- And 21 more...

Load with:
```python
model = gdrift.SeismicModel("3d_seismic_<model_name>")
```

#### Thermodynamic Models

| Name | Composition | Citation |
|------|-------------|----------|
| `SLB_21_pyroliteCFMAS` | Pyrolite CFMAS | Stixrude & Lithgow-Bertelloni (2021) |
| `SLB_21_pyroliteNCMAS` | Pyrolite NCMAS | Stixrude & Lithgow-Bertelloni (2021) |

Additional thermodynamic models (SLB_08, SLB_11, SLB_24) with various
compositions are available. Use `gdrift.mineralogy.MODELS_AVAIL` and
`gdrift.mineralogy.COMPOSITIONS_AVAIL` to see the full list.

#### Geodynamic Profiles

| Name | Description | Citation |
|------|-------------|----------|
| `1d_geodynamic_SLB21_pyroliteCFMAS` | Adiabatic profile for pyrolite | Computed from SLB_21 |

---

## Complete Citations

### Andrault et al. (2011)
Andrault, Denis, et al. "Solidus and liquidus profiles of chondritic mantle: Implication for melting of the Earth across its history." *Earth and Planetary Science Letters* 304.1-2 (2011): 251-259.

### Dziewonski & Anderson (1981)
Dziewonski, Adam M., and Don L. Anderson. "Preliminary reference Earth model." *Physics of the Earth and Planetary Interiors* 25.4 (1981): 297-356.

### Fiquet et al. (2010)
Fiquet, G., et al. "Melting of peridotite to 140 gigapascals." *Science* 329.5998 (2010): 1516-1518.

### French & Romanowicz (2014)
French, Scott W., and Barbara Romanowicz. "Whole-mantle radially anisotropic shear velocity structure from spectral-element waveform tomography." *Geophysical Journal International* 199.3 (2014): 1303-1327.

### Ghelichkhan et al. (2021)
Ghelichkhan, S., et al. "The adjoint method applied to time-dependent mantle convection." *Geophysical Journal International* (2021).

### Lei et al. (2020)
Lei, W., et al. "Global adjoint tomography—model GLAD-M25." *Geophysical Journal International* 223.1 (2020): 1-21.

### Nomura et al. (2014)
Nomura, Ryuichi, et al. "Low core-mantle boundary temperature inferred from the solidus of pyrolite." *Science* 343.6170 (2014): 522-525.

### Ritsema et al. (2011)
Ritsema, Jeroen, et al. "S40RTS: a degree-40 shear-velocity model for the mantle from new Rayleigh wave dispersion, teleseismic traveltime and normal-mode splitting function measurements." *Geophysical Journal International* 184.3 (2011): 1223-1236.

### Stixrude & Lithgow-Bertelloni (2011, 2016, 2021)
- Stixrude, L., and C. Lithgow-Bertelloni. "Thermodynamics of mantle minerals - I. Physical properties." *Geophysical Journal International* 162.2 (2005): 610-632.
- Stixrude, L., and C. Lithgow-Bertelloni. "Thermodynamics of mantle minerals - II. Phase equilibria." *Geophysical Journal International* 184.3 (2011): 1180-1213.
- Updates in 2016 and 2021 (see SLB website for details).

### Thrastarson et al. (2024)
Thrastarson, Solvi, et al. "REVEAL: A global full‐waveform inversion model." *Bulletin of the Seismological Society of America* 114.3 (2024): 1392-1406.

### Zerr & Boehler (1994)
Zerr, A., and R. Boehler. "Constraints on the melting temperature of the lower mantle from high-pressure experiments on MgO and magnesioüstite." *Nature* 371.6497 (1994): 506-508.

---

## Usage Examples

See the [Examples](examples/index.md) section for Jupyter notebooks demonstrating these APIs in action.
