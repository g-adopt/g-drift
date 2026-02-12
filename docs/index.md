---
hide:
  - navigation
  - toc
---

<div align="center">
  <img src="assets/images/logo.svg" alt="gdrift logo" width="300">
</div>

<div align="center">
  <strong>Geodynamics Data Reformatting and Integration Facilitation Toolkit</strong>
</div>

<br>

gdrift is a Python package for loading, processing, and converting geodynamic and seismic data used in large-scale mantle convection studies. It provides a unified interface for working with 1D reference Earth models, 3D seismic tomography models, thermodynamic lookup tables, and solidus profiles.

## Key Features

- **Unified Data Interface**: Single function to load 33+ curated datasets with automatic download and verification
- **1D Radial Profiles**: PREM, solidus profiles, and custom Earth models with spline interpolation
- **3D Seismic Models**: 25 tomography models with KD-tree interpolation and multiple kernel options
- **Thermodynamic Tables**: SLB (Stixrude & Lithgow-Bertelloni) pyroxene and basalt tables for mantle properties
- **Anelastic Corrections**: Cammarano, Goes, and Stixrude & Lithgow-Bertelloni models
- **Automatic Data Management**: Datasets stored on Digital Ocean Spaces with SHA256 verification

## Quick Start

### Installation

```bash
pip install gdrift
```

### Load a Dataset

```python
import gdrift

# Load PREM (1D reference model)
prem = gdrift.PreliminaryRefEarthModel()
depths = [100e3, 500e3, 1000e3]  # depths in meters
densities = prem.get_profile("rho").at_depth(depths)

# Load a solidus profile
solidus = gdrift.RadialEarthModelFromFile("1d_solidus_Andrault_et_al_2011_EPSL")
temps = solidus.get_profile("solidus temperature").at_depth(depths)

# Load a 3D seismic model
seismic = gdrift.SeismicModel("3d_seismic_S40RTS")
vs_anomaly = seismic.interpolate(lat=45.0, lon=10.0, depth=500e3)
```

### Temperature to Seismic Velocity

```python
# Load thermodynamic model
thermo = gdrift.ThermodynamicModel("SLB_16", "pyrolite")

# Convert temperature to seismic velocity
temperature = 1600  # Kelvin
depth = 500e3       # meters
vs = thermo.temperature_to_vs(temperature, depth)
vp = thermo.temperature_to_vp(temperature, depth)
rho = thermo.temperature_to_rho(temperature, depth)

# Inverse: velocity to temperature
temp_from_vs = thermo.vs_to_temperature(vs, depth)
```

### Apply Anelastic Corrections

```python
# Create anelastic model
anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile("Q3")

# Apply to thermodynamic model
corrected_thermo = gdrift.apply_anelastic_correction(thermo, anelastic)

# Now temperature_to_vs includes anelastic effects
vs_anelastic = corrected_thermo.temperature_to_vs(temperature, depth)
```

## Available Datasets

gdrift provides 33+ curated datasets organized into categories:

| Category | Count | Examples |
|----------|-------|----------|
| Reference Earth Models | 1 | PREM |
| Solidus Profiles | 6 | Andrault 2011, Fiquet 2010, Hirschmann 2000 |
| Seismic Tomography | 25 | S40RTS, GLAD-M25, SEMUCB-WM1, REVEAL |
| Thermodynamic Tables | 3 | SLB_16 pyrolite/basalt, SLB_21 pyrolite |
| Geodynamic Profiles | 1 | SLB21 adiabatic profile |

All datasets include:
- Automatic download from cloud storage (Digital Ocean Spaces)
- SHA256 hash verification on every load
- Proper citations and metadata
- Fallback to HTTPS when boto3 unavailable

## Documentation

- [Getting Started](getting-started.md) - Installation and basic usage
- [User Guide](user-guide/index.md) - Detailed documentation by topic
- [Examples](examples/index.md) - Jupyter notebook demonstrations
- [API Reference](api.md) - Complete API documentation
- [About](about.md) - Architecture and design decisions

## Citation

If you use gdrift in your research, please cite:

```bibtex
@software{gdrift2025,
  author = {Ghelichkhani, Sia},
  title = {gdrift: Geodynamics Data Reformatting and Integration Facilitation Toolkit},
  year = {2025},
  url = {https://github.com/g-adopt/g-drift}
}
```

Also please cite the relevant dataset sources - see [API Reference](api.md) for citations.

## Support

- **Issues**: [GitHub Issues](https://github.com/g-adopt/g-drift/issues)
- **Repository**: [github.com/g-adopt/g-drift](https://github.com/g-adopt/g-drift)
- **Documentation**: [gdrift.gadopt.org](https://gdrift.gadopt.org)

## License

MIT License - see LICENSE file for details.
