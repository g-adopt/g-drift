# Examples Directory - Jupytext and Testing Pipeline

# What python to use
All the python related commands should be using ~/Workplace/python3.12/bin/ for running python.
For gadopt/Firedrake testing locally: `/Users/sghelichkhani/Workplace/firedrake-2026-01-13/venv-firedrake/bin/python3`
with `PYTHONPATH=~/Workplace/g-adopt:~/Workplace/g-drift`

## Purpose

This directory contains Python scripts that serve as both executable examples and documentation. Following the pattern established in [gadopt demos](/Users/sghelichkhani/Workplace/g-adopt/demos/), these scripts are:

1. **Tested via pytest** - regression testing against pickled expected values
2. **Converted to Jupyter notebooks** - using jupytext for execution with outputs
3. **Rendered as documentation** - via mkdocs for the g-drift documentation site

## Demo Status

| Directory | Topic | Demo | Tests | Status |
|-----------|-------|------|-------|--------|
| `mantle_solidus/` | Loading and plotting solidus temperature profiles | ✅ | ✅ 5 tests | Complete |
| `anelasticity_corrections/` | Applying anelastic corrections to thermodynamic models | ✅ | ✅ tests | Complete |
| `geodynamic_adiabat/` | Computing adiabatic profiles with SLB_21 and SLB_24 models | ✅ | ✅ 9 tests | Complete |
| `linearisation/` | Regularising thermodynamic tables to smooth phase-transition jumps | ✅ | ✅ 8 tests | Complete |
| `gadopt_loading_field/` | Loading gdrift seismic data onto a gadopt mesh | ✅ | ✅ 8 tests (skip without gadopt) | Complete |
| `tomography_models/` | Loading and visualising 3D seismic tomography models | ✅ | **TODO** | Needs tests |
| `temperature_to_vs/` | Full Vs-to-temperature conversion pipeline (REVEAL + SLB_21) | ✅ | ✅ 7 tests | Complete |

## USER ACTION REQUIRED: Data Files to Move or Remove

The following data files/directories are in `examples/` but are not used by any demo. They may be
generated data from earlier work. Please move them somewhere appropriate or delete them:

- `geodynamic_profiles_filtered/` - directory with .txt files (alphabar, betabar, Cpbar, etc.)
- `geodynamic_profiles_filtered.tar.gz` - compressed version of the above
- `geodynamic_profiles_unfiltered.tar.gz` - another dataset archive

## Demos Still Needing Tests

### tomography_models/demo.py

Has `demo.py` and `Makefile` only. Demo script is complete with jupytext formatting. Needs:
1. `test_demo.py`, `expected.pkl`, `generate_expected.py`
2. Key values to test: number of global Vs models, dVs ranges for S40RTS/SEMUCB-WM1

## Required Files Per Demo

```
demo_name/
├── demo.py              # Main script (jupytext format)
├── test_demo.py         # Pytest test file
├── expected.pkl         # Pickled expected values
├── generate_expected.py # Utility to regenerate expected.pkl
└── Makefile             # Build targets (notebook, test, expected, clean)
```

## CI Integration

The pytest configuration includes both `tests/` and `examples/` directories:
```toml
[tool.pytest.ini_options]
testpaths = ["tests", "examples"]
```

The `test_firedrake.yml` workflow runs in a Firedrake container, installs gadopt, then runs:
```yaml
python -m pytest tests/ -v --maxfail=5 --tb=short
python -m pytest examples/ -v --maxfail=5 --tb=short
```

## File Structure

```
examples/
├── CLAUDE.md
├── Makefile                         # Top-level coordinator
├── TerraMT512vs.dat                 # Data file (used by gadopt_loading_field, temperature_to_vs)
├── ARCHIVE/                         # Stale/deprecated scripts (not tested)
├── mantle_solidus/                  # ✅ Complete (5 tests)
├── anelasticity_corrections/        # ✅ Complete
├── geodynamic_adiabat/              # ✅ Complete (9 tests)
├── linearisation/                   # ✅ Complete (8 tests)
├── gadopt_loading_field/            # ✅ Complete (8 tests, requires gadopt)
├── temperature_to_vs/               # ✅ Complete (7 tests)
├── tomography_models/               # Demo done, needs tests
├── geodynamic_profiles_filtered/    # USER: move or remove
├── geodynamic_profiles_filtered.tar.gz  # USER: move or remove
└── geodynamic_profiles_unfiltered.tar.gz  # USER: move or remove
```

## Jupytext Reference

| Marker | Purpose |
|--------|---------|
| `# +` | Start of code cell |
| `# -` | End of code cell |
| `# + tags=["active-ipynb"]` | Cell only appears in notebook |
| `# + tags=["remove-input"]` | Hide input in rendered notebook |

RST formatting: `# Title` with `# =====` underline, `# Section` with `# -----`, `# $LaTeX$` for math.

## API Gotchas

- `gdrift.geodetic_to_cartesian(lat, lon, depth)` returns **Nx3 array** (not a tuple)
- `gdrift.cartesian_to_geodetic(x, y, z)` returns **tuple of arrays**
- `model.at()` expects coordinates as Nx3 array
- Mollweide projection requires coordinates in **radians**
- `gadopt_loading_field` and `temperature_to_vs` demos require `TerraMT512vs.dat` in parent `examples/` directory
- When using `exec()` to run demo.py in tests, inject `__file__` into the namespace if the demo uses `Path(__file__)` (see `temperature_to_vs/test_demo.py` for example)

## Known Issues

### SLB_24 upper-mantle velocity overprediction
The SLB_24 pyrolite (CFMS) model significantly overpredicts absolute shear-wave velocities in the
upper mantle compared to seismic observations (e.g., REVEAL). At 200 km depth and adiabatic
temperatures, SLB_24 gives Vs ≈ 5400 m/s while REVEAL observes ≈ 4540 m/s — a ~19% mismatch.
This leads to non-physical temperatures (2600–3300 K) when inverting Vs → T at shallow depths.

SLB_21 (pyroliteCFMAS) does **not** have this problem: at 200 km its corrected Vs (≈ 4578 m/s)
matches REVEAL (≈ 4541 m/s) to within 1%. The match remains good throughout the mantle (< 3%
mismatch at most depths). **Use SLB_21 for Vs-to-temperature inversions until the SLB_24 issue
is resolved.**

### Reference temperature profile for regularisation
A pure adiabat (e.g., from `compute_adiabat(T0=1600)`) is not a good regularisation anchor for the
full mantle because it misses both thermal boundary layers:
- **Upper (lithospheric)**: temperature rises from ~300 K at the surface to the adiabat over
  ~100–150 km.
- **Lower (D″ layer)**: temperature rises from the adiabat to ~4200 K at the CMB over ~200 km.

The `TerraMT512vs.dat` file (azimuthally-averaged profile from a Terra mantle convection simulation)
captures both boundary layers and should be used as the regularisation anchor. Column 0 is depth
in km, column 1 is temperature in K. An alternative approach (not yet implemented) would be to
construct a synthetic profile by splicing error-function boundary layers onto a computed adiabat.

### NaN values in ThermodynamicModel property queries
Some SLB_21 models produce NaN for in-range queries. Workaround: `compute_adiabat()` clamps
depth/temperature to table bounds. Works reliably with `SLB_21_pyroliteCFMAS` and `SLB_24_pyroliteCFMS`.

### Surface-based dissipation number is ~1.6
The surface-value Di is higher than the commonly cited 0.5-0.7 (which uses depth-averaged properties).
