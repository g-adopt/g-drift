# About gdrift

## Overview

gdrift (Geodynamics Data Reformatting and Integration Facilitation Toolkit) provides a unified Python interface for working with diverse geodynamic and seismic datasets. The package was designed to simplify the process of loading, interpolating, and converting between different representations of Earth structure used in mantle convection modeling.

## Design Philosophy

### Single Source of Truth

All dataset metadata, download URLs, and verification hashes are stored in a single JSON manifest (`gdrift/datasets.json`) that ships with the package. This ensures:

- **Reproducibility**: Every installation uses the same dataset versions
- **Integrity**: SHA256 hashes verify file authenticity on every load
- **Transparency**: All dataset sources and citations are documented
- **Maintainability**: Adding new datasets requires only updating one file

### Automatic Data Management

gdrift handles the complexity of data download and caching:

1. **On-demand download**: Datasets are fetched only when first accessed
2. **Cloud storage**: Primary hosting on Digital Ocean Spaces (S3-compatible)
3. **HTTPS fallback**: Automatic fallback to CDN if boto3 unavailable
4. **Hash verification**: Every load checks file integrity against manifest
5. **Auto-recovery**: Corrupted files trigger automatic re-download

### Physics-Aware APIs

Rather than exposing raw HDF5 arrays, gdrift provides domain-specific classes:

- `RadialEarthModel`: Knows about Earth's structure (core-mantle boundary, surface)
- `ThermodynamicModel`: Understands temperature-velocity-density relationships
- `SeismicModel`: Provides geographic interpolation with appropriate kernels
- `AnelasticityModel`: Encapsulates frequency-dependent attenuation physics

## Architecture

### Data Pipeline

```
datasets.json (manifest)
        ↓
DatasetRegistry (runtime)
        ↓
load_dataset() (download + verify)
        ↓
Domain Classes (physics-aware APIs)
```

1. **Manifest** (`datasets.json`): JSON file containing S3 config, CDN URL, and 33 dataset entries with metadata
2. **Registry** (`datasetnames.py`): Loads manifest at import time, builds `Dataset` dataclass instances
3. **I/O** (`io.py`): Validates dataset name, checks cache, downloads via boto3/HTTPS, verifies SHA256
4. **Domain Classes**: Consume raw arrays and expose interpolation/conversion APIs

### Module Organization

| Module | Purpose |
|--------|---------|
| `datasets.json` | Canonical manifest: S3 config, CDN URL, all dataset entries with SHA256 hashes |
| `io.py` | Download (boto3 S3 + HTTPS fallback), cache, hash-verify, and load HDF5 datasets |
| `datasetnames.py` | Dataset registry built from manifest; enums, dataclasses, query helpers |
| `profile.py` | 1D radial profiles with spline interpolation |
| `earthmodel3d.py` | 3D KD-tree interpolation with multiple kernel options |
| `seismic.py` | 3D seismic tomography models (inherits `EarthModel3D`) |
| `mineralogy.py` | Thermodynamic tables, temperature-property conversions |
| `anelasticity.py` | Anelastic corrections (Cammarano, Goes, Stixrude & Lithgow-Bertelloni) |
| `utility.py` | Coordinate transforms, gravity/pressure computation |
| `constants.py` | Physical constants (Earth radius, CMB radius, etc.) |
| `gplates/coastlines.py` | PyGPlates integration for coastline reconstruction |

### Class Hierarchy

```
AbstractEarthModel (ABC)
  └── EarthModel3D          (KD-tree + kernel interpolation)
        └── SeismicModel     (loads 3D tomography HDF5)

AbstractProfile (ABC)
  ├── SplineProfile          (1D spline interpolation)
  └── HirschmannSolidusProfile (depth → pressure → solidus T)

RadialEarthModel             (composite of multiple profiles)
  ├── RadialEarthModelFromFile (loads profiles from HDF5)
  │     └── PreliminaryRefEarthModel (PREM)
  └── HirschmannSolidus

ThermodynamicModel           (2D depth×temperature tables)
  └── RegularisedThermodynamicModel (dynamic phase transition smoothing)

BaseAnelasticityModel (ABC)
  ├── CammaranoAnelasticityModel   (B, g, a parameterization)
  └── GoesAnelasticityModel        (Q0, xi parameterization)
```

## Key Design Decisions

### Why boto3 + JSON Manifest?

**Problem**: Previous approach used hardcoded URLs and Pooch for downloads, making it difficult to update datasets or add new ones.

**Solution**:
- JSON manifest as single source of truth
- boto3 for efficient S3 access (unsigned public buckets)
- HTTPS fallback ensures availability even without boto3
- SHA256 hashes prevent data corruption

**Benefits**:
- Add datasets by editing one JSON file
- Automatic hash verification on every load
- Works offline if data already cached
- Cloud storage scales to terabytes

### Why KD-Tree Interpolation?

**Problem**: 3D seismic models can have millions of points; nearest-neighbor search must be fast.

**Solution**: scipy's KDTree for O(log N) nearest-neighbor queries.

**Benefits**:
- Fast interpolation at arbitrary lat/lon/depth points
- Multiple kernel options (IDW, Gaussian, Wendland, linear)
- Supports both geographic and Cartesian coordinates
- Memory-efficient for large models

### Why Dataclasses for Registry?

**Problem**: String-based dataset names are error-prone; need type safety and IDE autocomplete.

**Solution**: Generate `Dataset` dataclass instances from manifest with typed fields.

**Benefits**:
- Type hints for dataset metadata (name, type, utility, citation)
- Query helpers (filter by type, search by keyword)
- Validation at load time (reject unknown datasets)
- IDE autocomplete for dataset names

### Why Separate Anelasticity from Thermodynamics?

**Problem**: Anelastic corrections depend on temperature, frequency, and rheology - they're conceptually separate from elastic properties.

**Solution**: Separate `AnelasticityModel` classes that wrap `ThermodynamicModel` instances.

**Benefits**:
- Clean separation of concerns
- Multiple anelastic models can wrap the same thermo model
- User controls whether to apply corrections
- Easy to compare elastic vs. anelastic predictions

## Interpolation Kernels

gdrift supports multiple interpolation kernels for 3D models:

| Kernel | Formula | Use Case |
|--------|---------|----------|
| IDW (Inverse Distance Weighting) | $w_i = 1/d_i^p$ | General purpose, adjustable smoothness |
| Gaussian | $w_i = \exp(-d_i^2 / 2\sigma^2)$ | Smooth interpolation |
| Wendland | $w_i = (1 - d_i/h)^4 (4d_i/h + 1)$ | Compact support, fast |
| Linear | Linear barycentric | Fast but less accurate |
| Nearest | $w_i = \delta_{i,\text{nearest}}$ | Fastest, no smoothing |

Set kernel with:
```python
seismic.set_kernel(kernel="gaussian", sigma=100e3)  # 100 km smoothing
```

## Thermodynamic Models

gdrift uses lookup tables from Stixrude & Lithgow-Bertelloni (SLB) for mineral physics:

- **SLB_21**: Pyrolite CFMAS and NCMAS (2021 parameterization)
- Additional models (SLB_08, SLB_11, SLB_24) available via the manifest

Tables are 2D grids in (depth, temperature) space with properties:
- Density (`rho`)
- Shear modulus (`shear_mod`)
- Bulk modulus (`bulk_mod`)
- S-wave velocity (`v_s`)
- P-wave velocity (`v_p`)
- And more...

### Phase Transition Regularization

The `regularise_thermodynamic_table()` function smooths sharp phase transitions:

```python
from gdrift import regularise_thermodynamic_table

# Smooth 410 km and 660 km discontinuities
smoothed = regularise_thermodynamic_table(
    thermo_model,
    depths_to_regularise=[410e3, 660e3],
    regularisation_widths=[20e3, 30e3]
)
```

This is essential for finite-element modeling where sharp discontinuities can cause convergence issues.

## Anelasticity Models

gdrift implements three published anelasticity models:

### Cammarano et al. (2003)

Parameterized by:
- $B(T)$: Activation energy ratio
- $g(T)$: Power-law exponent
- $a(T)$: Attenuation factor

Six preset Q-profiles (Q1-Q6) from different studies.

### Goes et al. (2004)

Parameterized by:
- $Q_0$: Reference quality factor
- $\xi$: Temperature sensitivity

Two preset profiles: Q4 and Q6.

### Stixrude & Lithgow-Bertelloni (2005)

Built into SLB thermodynamic models; frequency-dependent.

## Coordinate Systems

gdrift supports two coordinate systems:

### Geographic (lat, lon, depth)
- Latitude: -90° (South) to +90° (North)
- Longitude: -180° (West) to +180° (East)
- Depth: meters below surface (positive downward)

### Cartesian (x, y, z)
- Origin at Earth's center
- z-axis through North Pole
- x-axis through 0° latitude, 0° longitude

Convert between them:
```python
import gdrift

x, y, z = gdrift.geodetic_to_cartesian(lat, lon, depth)
lat, lon, depth = gdrift.cartesian_to_geodetic(x, y, z)
```

## Performance Considerations

### Caching
- Datasets cached in `~/.cache/gdrift/data/`
- Subsequent loads are fast (no re-download)
- Cache persists across Python sessions

### KD-Tree Build Time
- 3D seismic models build KD-tree on first interpolation (~1-10 seconds for large models)
- Tree is reused for subsequent queries
- Memory footprint: ~2-3× raw data size

### Thermodynamic Table Lookups
- 2D interpolation via `scipy.interpolate.RegularGridInterpolator`
- Fast lookups: ~10-100 μs per query
- Vectorized operations supported

## Dataset Citations

When using gdrift datasets, please cite the original sources:

### PREM
Dziewonski, A. M., & Anderson, D. L. (1981). Preliminary reference Earth model. *Physics of the Earth and Planetary Interiors*, 25(4), 297-356.

### S40RTS
Ritsema, J., et al. (2011). S40RTS: a degree-40 shear-velocity model for the mantle from new Rayleigh wave dispersion. *Geophysical Journal International*, 184(3), 1223-1236.

### SLB Thermodynamics
Stixrude, L., & Lithgow-Bertelloni, C. (2011). Thermodynamics of mantle minerals - II. Phase equilibria. *Geophysical Journal International*, 184(3), 1180-1213.

See the [API Reference](api.md) for complete citation information for all datasets.

## Contributing

Contributions are welcome! To add a new dataset:

1. Add entry to `gdrift/datasets.json` with metadata and SHA256 hash
2. Upload HDF5 file to Digital Ocean Spaces (or provide URL)
3. Add utility class if needed (e.g., new seismic model type)
4. Add tests in `tests/`
5. Update documentation
6. Submit pull request

See the [GitHub repository](https://github.com/sghelichkhani/g-drift) for development guidelines.

## License

MIT License - see [LICENSE](https://github.com/sghelichkhani/g-drift/blob/main/LICENSE) file for details.
