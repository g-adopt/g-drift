---
hide:
  - navigation
  - toc
---

<div align="center">
  <img src="assets/images/logo.svg" alt="gdrift logo" width="250">
</div>

<div align="center" style="margin-top: 1em;">
  <strong style="font-size: 1.2em;">A community platform for accessing geodynamic data</strong>
</div>

<br>

**gdrift** provides a unified Python interface for loading geodynamic and seismic datasets used in mantle convection studies — 1D reference profiles, 3D tomography models, thermodynamic lookup tables, and solidus temperatures. All datasets are hosted remotely and downloaded on demand with integrity verification.

```bash
pip install gdrift
```

```python
import gdrift

prem = gdrift.PreliminaryRefEarthModel()
thermo = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
seismic = gdrift.SeismicModel("3d_seismic_S40RTS")
```

Browse the [examples](examples/index.md) to see what you can do, or check the [data catalog](data-catalog.md) to see what's available.

---

<div style="margin-top: 2em; font-size: 0.9em; color: #555;">

**Contact:** Siavash Ghelichkhan — [siavash.ghelichkhan@anu.edu.au](mailto:siavash.ghelichkhan@anu.edu.au) — [sia-g.com](https://sia-g.com)

If you think something is wrong with this, you can be part of fixing it: [github.com/g-adopt/g-drift](https://github.com/g-adopt/g-drift)

</div>
