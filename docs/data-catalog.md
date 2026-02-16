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

When using these datasets, please cite both the original publication (DOI links below) and the [gdrift package](https://github.com/g-adopt/g-drift).

---

--8<-- "data-catalog-generated.md"
