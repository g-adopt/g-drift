# Loading and Querying a Seismic Tomography Model
# =================================================
#
# This example demonstrates how to load a 3D seismic tomography model
# and query it at arbitrary points using gdrift. We use S40RTS
# (Ritsema et al., 2011), one of the most widely-used global shear-wave
# velocity models, as a concrete example.
#
# Background
# ----------
#
# Seismic tomography is a technique for imaging Earth's interior by analysing
# differences in seismic wave travel times. Faster velocities (positive dVs/Vs)
# typically indicate colder material, while slower velocities (negative dVs/Vs)
# suggest hotter regions. These velocity anomalies are key inputs for inferring
# mantle temperature structure in geodynamic studies.
#
# The gdrift package provides access to 47 published tomography models,
# including global and regional models with shear-wave (dVs) and/or
# compressional-wave (dVp) velocity perturbations. A visual overview of
# all models is available in the `Data Catalog <../data-catalog.md>`_.
#
# This example
# ------------
#
# We demonstrate how to:
# 1. List all available seismic tomography models
# 2. Load S40RTS and inspect its properties
# 3. Query the model at a depth slice (2700 km, near the core-mantle boundary)
# 4. Visualise the dVs field on a Mollweide projection

import numpy as np
import gdrift

# Available Models
# ----------------
#
# The full list of tomography models is built from the dataset manifest
# at import time. Each entry corresponds to an HDF5 file on the server
# that is downloaded on first use.

# +
print(f"Number of available models: {len(gdrift.AVAILABLE_SEISMIC_MODELS)}")
print("\nFirst 10 models:")
for name in gdrift.AVAILABLE_SEISMIC_MODELS[:10]:
    print(f"  - {name}")
# -

# Loading S40RTS
# --------------
#
# We load S40RTS by passing its name to ``SeismicModel``. This downloads
# the HDF5 file (if not already cached), reads all fields, and builds a
# KD-tree for spatial interpolation.

# +
model = gdrift.SeismicModel("S40RTS")

print(f"Model: S40RTS")
print(f"Available fields: {list(model.available_fields.keys())}")
print(f"Number of data points: {len(model.coordinates)}")
# -

# Inspecting Model Extent
# -----------------------
#
# Each model stores its data points in Cartesian coordinates (x, y, z in
# metres from Earth's centre). We convert to geographic coordinates to
# inspect the spatial coverage.

# +
coords = model.coordinates
lat, lon, depth = gdrift.cartesian_to_geodetic(
    coords[:, 0], coords[:, 1], coords[:, 2]
)

print(f"Latitude range:  {lat.min():.1f} to {lat.max():.1f} degrees")
print(f"Longitude range: {lon.min():.1f} to {lon.max():.1f} degrees")
print(f"Depth range:     {depth.min()/1e3:.0f} to {depth.max()/1e3:.0f} km")
# -

# Querying at a Fixed Depth
# -------------------------
#
# We create a regular grid in latitude and longitude at 2700 km depth,
# close to the core-mantle boundary. This depth is of particular interest
# because it reveals Large Low Shear Velocity Provinces (LLSVPs) beneath
# Africa and the Pacific.
#
# The ``at()`` method accepts an Nx3 array of Cartesian coordinates and
# returns interpolated values using inverse distance weighting over the
# 8 nearest data points.

# +
depth_km = 2700
depth_m = depth_km * 1e3

lats = np.arange(-90, 91, 1)
lons = np.arange(-180, 181, 1)
lon_grid, lat_grid = np.meshgrid(lons, lats)
depth_grid = np.full_like(lat_grid, depth_m, dtype=float)

query_coords = gdrift.geodetic_to_cartesian(
    lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel()
)

dvs = model.at("dvs", query_coords)
dvs_grid = dvs.reshape(lat_grid.shape)

print(f"Query grid: {len(lats)} x {len(lons)} = {query_coords.shape[0]} points")
print(f"dVs range at {depth_km} km: {np.nanmin(dvs_grid):.2f}% to {np.nanmax(dvs_grid):.2f}%")
# -

# Visualisation
# -------------
#
# We plot the dVs field on a Mollweide projection using a diverging
# colour scale centred at zero. Red regions (negative dVs) indicate
# slower-than-average velocities (hotter material), while blue regions
# (positive dVs) indicate faster-than-average velocities (colder material).
#
# The two prominent red patches beneath Africa and the Pacific are the
# LLSVPs, thermochemical structures that are among the largest features
# in the deep mantle.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
#
# alpha = np.nanmax(np.abs(dvs_grid))
# norm = TwoSlopeNorm(vmin=-alpha, vcenter=0, vmax=alpha)
#
# fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "mollweide"})
# im = ax.pcolormesh(
#     np.radians(lon_grid), np.radians(lat_grid), dvs_grid,
#     cmap="RdBu", norm=norm, shading="auto",
# )
# ax.grid(True, alpha=0.3)
# ax.set_title(f"S40RTS at {depth_km} km depth", fontsize=14, fontweight="bold")
#
# cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
# cbar.set_label(r"$\delta V_s / V_s$ (%)", fontsize=11)
#
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example showed how to:
#
# - List available models via ``gdrift.AVAILABLE_SEISMIC_MODELS``
# - Load a model with ``gdrift.SeismicModel("S40RTS")``
# - Inspect fields (``available_fields``) and coordinate extent
# - Query at arbitrary points with ``model.at("dvs", coordinates)``
# - Visualise a depth slice on a Mollweide projection
#
# The same workflow applies to any of the 47 available models. See the
# `Data Catalog <../data-catalog.md>`_ for a gallery of all models.
