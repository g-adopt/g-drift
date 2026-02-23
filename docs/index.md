---
title: ""
hide:
  - navigation
  - toc
---

<div align="center">
  <img src="assets/images/logo.svg" alt="gdrift logo" width="250">
</div>

<div align="center" style="margin-top: 1em;">
  <strong style="font-size: 1.2em;">Data arm of G-ADOPT for FAIR geodynamic inverse modelling</strong>
</div>

<br>

**GDRIFT** provides a unified Python interface for loading geodynamic and seismic datasets used in mantle convection studies. Browse our full collection of datasets and examples in the [Data Catalogue](datasets.md).

!!! warning "Community Effort for FAIR Geodynamic Data"

    This is a community-driven initiative to create a modular, unified package for accessing geodynamic data — usable by anyone, not only G-ADOPT. All datasets presented here have been published elsewhere; we've invested effort to unify them in one place to enable reproducibility and FAIR principles.

    If you are a data author who feels there is good reason for your data not to be included here, please reach out to [siavash.ghelichkhan@anu.edu.au](mailto:siavash.ghelichkhan@anu.edu.au) or visit [sia-g.org](https://sia-g.org) to contact us.

```bash
pip install gdrift
```

```python
import gdrift

prem = gdrift.PreliminaryRefEarthModel()
thermo = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
seismic = gdrift.SeismicModel("3d_seismic_S40RTS")
```

!!! info "Zenodo Archival & Dataset Contributions"

    **Reproducible Versions**: GDRIFT is integrating with Zenodo to archive every major version. This ensures that if you have a script running against a specific version of GDRIFT, the datasets and functionality remain documented and accessible indefinitely.

    **Have a Dataset?** If you have a geodynamic dataset you'd like to contribute to this community—whether for G-ADOPT or any other application—we'd be happy to help integrate it. Reach out to [siavash.ghelichkhan@anu.edu.au](mailto:siavash.ghelichkhan@anu.edu.au) or visit [sia-g.org](https://sia-g.org).

---

<div style="margin-top: 2em; font-size: 0.9em; color: #666;">

Found an issue or want to contribute? See the [source code on GitHub](https://github.com/g-adopt/g-drift).

</div>
