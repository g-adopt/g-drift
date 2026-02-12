# Getting Started

This guide will help you install gdrift and get started with basic usage.

## Installation

### Requirements

- Python 3.12 or later
- pip package manager

### Basic Installation

Install gdrift from PyPI:

```bash
pip install gdrift
```

This installs the core dependencies:
- `numpy` - Numerical arrays
- `scipy` - Scientific computing (spline interpolation, KD-tree)
- `h5py` - HDF5 file reading
- `tqdm` - Progress bars for downloads
- `boto3` - S3-compatible cloud storage access

### Optional Dependencies

For running examples and visualization:

```bash
pip install gdrift[examples]
```

This adds:
- `matplotlib` - Plotting
- `jupytext` - Python-to-notebook conversion
- `nbconvert` - Notebook execution
- `ipykernel` - Jupyter kernel

For development (testing, linting):

```bash
pip install gdrift[dev]
```

For documentation building:

```bash
pip install gdrift[docs]
```

Install everything:

```bash
pip install gdrift[all]
```

### Development Installation

To install from source for development:

```bash
git clone https://github.com/sghelichkhani/g-drift.git
cd g-drift
pip install -e ".[all]"
```

The `-e` flag installs in editable mode, so changes to the source code are immediately reflected.

## First Steps

### Verify Installation

```python
import gdrift
print(gdrift.__version__)
```

### Data Storage

gdrift automatically downloads datasets on first use to:

```
~/.cache/gdrift/data/
```

Or on macOS:
```
~/Library/Caches/gdrift/data/
```

You can change this by setting the `GDRIFT_DATA_DIR` environment variable:

```bash
export GDRIFT_DATA_DIR=/path/to/your/data
```

### Load Your First Dataset

```python
import gdrift

# Load PREM (Preliminary Reference Earth Model)
prem = gdrift.PreliminaryRefEarthModel()

# Get density profile
rho_profile = prem.get_profile("rho")

# Evaluate at specific depths (in meters)
depths = [100e3, 500e3, 1000e3, 2000e3]  # 100, 500, 1000, 2000 km
densities = rho_profile.at_depth(depths)

print(f"Densities: {densities} kg/m³")
```

Output:
```
Densities: [3381.97 3689.37 4441.09 5055.78] kg/m³
```

### Load a Seismic Tomography Model

```python
import gdrift

# Load S40RTS tomography model
# On first run, this downloads ~200 MB from cloud storage
seismic = gdrift.SeismicModel("3d_seismic_S40RTS")

# Interpolate at a point
lat, lon, depth = 45.0, 10.0, 500e3  # degrees, degrees, meters
vs_anomaly = seismic.interpolate(lat, lon, depth)

print(f"Vs anomaly at ({lat}°, {lon}°, {depth/1e3} km): {vs_anomaly:.2f}%")
```

### Load a Thermodynamic Model

```python
import gdrift

# Load SLB (Stixrude & Lithgow-Bertelloni) pyrolite model
thermo = gdrift.ThermodynamicModel("SLB_16", "pyrolite")

# See available properties
print(thermo.available_tables())
# Output: ['bulk_mod', 'rho', 'shear_mod', 'v_s', 'v_p', ...]

# Convert temperature to seismic velocity
temperature = 1600  # Kelvin
depth = 500e3       # meters
vs = thermo.temperature_to_vs(temperature, depth)
vp = thermo.temperature_to_vp(temperature, depth)

print(f"Vs: {vs:.2f} m/s, Vp: {vp:.2f} m/s")
```

## Next Steps

### User Guide

Learn about specific functionality:

- [Datasets](user-guide/datasets.md) - Complete dataset catalog
- [1D Profiles](user-guide/profiles.md) - Radial Earth models
- [3D Seismic Models](user-guide/seismic.md) - Tomography interpolation
- [Thermodynamics](user-guide/thermodynamics.md) - Temperature-velocity conversions
- [Anelasticity](user-guide/anelasticity.md) - Anelastic corrections

### Examples

Work through interactive Jupyter notebooks:

- [Solidus Temperature](examples/00_solidus.ipynb)
- [Anelasticity Models](examples/01_anelasticity.ipynb)
- [Temperature to Vs Conversion](examples/02_temperature_to_vs.ipynb)
- [Tomography Models](examples/04_tomography.ipynb)
- [Geodynamic Adiabat](examples/05_geodynamic_adiabat.ipynb)

### API Reference

Browse the complete [API Reference](api.md) for detailed documentation of all classes and functions.

## Troubleshooting

### Dataset Download Fails

If boto3 is unavailable or S3 access fails, gdrift automatically falls back to HTTPS download via CDN:

```
https://gadopt.nyc3.cdn.digitaloceanspaces.com/g-drift/
```

### SHA256 Verification Fails

If a dataset file is corrupted, gdrift automatically re-downloads it. If the problem persists, manually delete the cached file:

```bash
rm ~/.cache/gdrift/data/<dataset_name>.h5
```

### Import Errors

Ensure you're using Python 3.12 or later:

```bash
python --version
```

If using a virtual environment, make sure it's activated:

```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

## Getting Help

- Check the [User Guide](user-guide/index.md) for detailed documentation
- Browse [Examples](examples/index.md) for working code
- Search [GitHub Issues](https://github.com/sghelichkhani/g-drift/issues)
- Open a new issue for bugs or feature requests
