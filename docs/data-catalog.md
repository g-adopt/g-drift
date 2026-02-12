# Data Catalog

Welcome to the comprehensive data catalog for gdrift. This page provides complete citation information and metadata for all datasets available in the package.

## About This Catalog

The gdrift package provides access to 96+ scientific datasets spanning:

- **Reference Earth Models**: 1D radial profiles (PREM)
- **Solidus Profiles**: Experimental melting temperatures (Andrault, Fiquet, Hirschmann)
- **Geodynamic Profiles**: Adiabatic temperature and property profiles
- **Thermodynamic Models**: 2D lookup tables for mineral physics (SLB database)
- **Seismic Tomography Models**: 3D velocity perturbation models

All datasets are:

- Downloaded on-demand from cloud storage (Digital Ocean Spaces)
- Verified via SHA256 hash checking
- Cached locally after first use
- Fully documented with complete citations

## Using Datasets

Each dataset can be loaded using its dedicated utility class:

```python
import gdrift

# Load a reference Earth model
prem = gdrift.PreliminaryRefEarthModel()

# Load a solidus profile
solidus = gdrift.RadialEarthModelFromFile("1d_solidus_Andrault_et_al_2011_EPSL")

# Load a thermodynamic model
thermo = gdrift.ThermodynamicModel("SLB_21", "pyrolite")

# Load a seismic tomography model
seismic = gdrift.SeismicModel("S40RTS")
```

## Citation Guidelines

When using gdrift datasets in your research, please cite:

1. **The original dataset publication** (DOI links provided below)
2. **The gdrift package itself**

Example citation format:
```
This research used the S40RTS seismic tomography model (Ritsema et al., 2011, DOI: 10.1111/j.1365-246X.2010.04884.x)
via the gdrift package (https://github.com/g-adopt/g-drift).
```

## Dataset Categories

The catalog below is organized by dataset type. For datasets still needing complete metadata, see the [incomplete metadata list](data-catalog-incomplete.md).

---

## Complete Dataset Listings

<!-- Auto-generated content via pymdownx.snippets -->
--8<-- "data-catalog-generated.md"

---

## Datasets Needing Metadata

A small number of datasets have incomplete citation information. Engineers working on metadata enrichment can find the list of incomplete entries in [data-catalog-incomplete.md](data-catalog-incomplete.md).

## Metadata Sources

Dataset metadata is extracted from multiple sources:

- **HDF5 files**: File-level attributes stored in thermodynamic model files
- **SLB references**: Standard citations for Stixrude & Lithgow-Bertelloni database versions
- **Web search**: DOI lookup for seismic tomography models
- **Manual curation**: Hand-verified citations and descriptions

## Notes

- This catalog is auto-generated in CI from `gdrift/datasets.json`
- Metadata is enriched via the `scripts/enrich_datasets_metadata.py` pipeline
- Generated markdown files are not tracked in git (regenerated on every build)
- SHA256 hashes ensure data integrity for all downloads

## Questions or Issues?

If you notice incorrect or incomplete metadata, please:

1. Check the [incomplete list](data-catalog-incomplete.md) first
2. Open an issue at [github.com/g-adopt/g-drift/issues](https://github.com/g-adopt/g-drift/issues)
3. Provide the dataset name and correct citation information
