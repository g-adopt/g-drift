# Data Catalog

All datasets are downloaded on demand and verified via SHA256 hashes.

```python
import gdrift

# Reference Earth model
prem = gdrift.PreliminaryRefEarthModel()

# Solidus profile
solidus = gdrift.RadialEarthModelFromFile("1d_solidus_Andrault_et_al_2011_EPSL")

# Thermodynamic model
thermo = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")

# Seismic tomography model
seismic = gdrift.SeismicModel("3d_seismic_S40RTS")
```

## Citations

When using datasets from gdrift, please cite both the original publication and the gdrift package. Each dataset entry below includes the original citation with a DOI link where available. In addition, please cite:

Ghelichkhan, S. (2025). gdrift: Geodynamics Data Reformatting and Integration Facilitation Toolkit. [github.com/g-adopt/g-drift](https://github.com/g-adopt/g-drift)

The metadata for all datasets, including citation information, is maintained in the `datasets.json` manifest that ships with the package. If you notice any missing or incorrect citation, please open an issue on the [GitHub repository](https://github.com/g-adopt/g-drift/issues).

---

--8<-- "tomography-gallery-generated.md"

---

--8<-- "data-catalog-generated.md"
