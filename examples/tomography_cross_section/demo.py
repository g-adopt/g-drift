# Plotting a Tomography Cross-Section
# ====================================
#
# This example demonstrates how to plot a great-circle cross-section
# through a seismic tomography model. While depth slices (covered in the
# `Seismic Tomography <../tomography_models/demo.ipynb>`_ example) show
# lateral velocity variations at a single depth, cross-sections reveal
# how anomalies extend radially through the mantle. This is particularly
# useful for visualising subducting slabs, mantle plumes, and Large Low
# Shear Velocity Provinces (LLSVPs).
#
# We use the REVEAL model (Thrastarson et al., 2024), a global
# full-waveform inversion model that provides density, compressional-wave
# and shear-wave velocities on a high-resolution global grid.
#
# Background
# ----------
#
# A great-circle cross-section samples the mantle along a plane that
# passes through the centre of the Earth. By choosing appropriate
# endpoints we can slice through features of interest. In this example
# we follow an arc from the equatorial mid-Atlantic Ridge to the Afar
# triple junction, cutting through the African LLSVP.
#
# Because REVEAL stores absolute velocities (in m/s) rather than
# perturbations, we need to remove the laterally averaged signal at
# each depth to isolate lateral variations. Subtracting the row-mean
# at each depth level produces a field analogous to
# $\delta V_s / \bar{V}_s(z)$ without requiring an external
# reference model.
#
# This example
# ------------
#
# We demonstrate how to:
# 1. Load the REVEAL tomography model and inspect its fields
# 2. Build a great-circle cross-section grid using
#    ``gdrift.great_circle_cross_section``
# 3. Compute the isotropic shear-wave velocity via the Voigt average
# 4. Remove the depth-by-depth lateral mean to obtain perturbations
# 5. Plot the result on a polar wedge projection

import numpy as np
import gdrift
from gdrift.constants import R_earth, R_cmb

# Loading REVEAL
# --------------
#
# We load the REVEAL model by passing its name to ``SeismicModel``. The
# HDF5 file is downloaded automatically on first use and cached locally.

# +
model = gdrift.SeismicModel("REVEAL")

print("Model: REVEAL")
print(f"Available fields: {list(model.available_fields.keys())}")
print(f"Number of data points: {len(model.coordinates)}")
# -

# REVEAL provides ``vsv``, ``vsh``, ``vpv``, and ``rho``. Since it stores
# anisotropic shear-wave components separately, we will compute the
# isotropic Vs using the Voigt average:
#
# $$V_s^{\text{Voigt}} = \sqrt{\frac{2\,V_{sv}^2 + V_{sh}^2}{3}}$$

# Building the Cross-Section Grid
# --------------------------------
#
# ``gdrift.great_circle_cross_section`` builds a 2-D sampling grid along
# a great-circle arc between two surface points. It returns:
#
# - ``theta_grid``: angular distance from the arc midpoint (radians)
# - ``r_grid``: radial distance from Earth's centre (metres)
# - ``query_coords``: Cartesian coordinates suitable for ``model.at()``
# - ``arc_info``: metadata about the arc geometry
#
# We choose endpoints that cross the African LLSVP: from the equatorial
# mid-Atlantic Ridge (0N, 30W) to the Afar triple junction (11.5N, 42E).

# +
lat_A, lon_A = 0.0, -30.0     # equatorial mid-Atlantic Ridge
lat_B, lon_B = 11.5, 42.0     # Afar triple junction
n_arc = 360                    # points along the arc
n_depth = 60                   # depth levels

theta_grid, r_grid, query_coords, arc_info = gdrift.great_circle_cross_section(
    lat_A=lat_A, lon_A=lon_A,
    lat_B=lat_B, lon_B=lon_B,
    n_arc=n_arc, n_depth=n_depth,
    min_depth=50e3,
)

arc_deg = np.degrees(arc_info["arc_length"])
print(f"Arc length: {arc_deg:.0f} degrees")
print(f"Query grid: {n_depth} depths x {n_arc} arc points = {query_coords.shape[0]} points")
# -

# Querying Velocities and Computing Perturbations
# -------------------------------------------------
#
# We query the two shear-wave components along the cross-section, compute
# the Voigt-averaged isotropic Vs, then subtract the row-mean at each
# depth to isolate lateral variations. This is the standard approach for
# absolute-velocity models: the row mean acts as a depth-dependent
# reference, so the residual highlights where velocities are anomalously
# fast (cold material) or slow (hot material).

# +
vsv = model.at("vsv", query_coords)
vsh = model.at("vsh", query_coords)
vs_voigt = np.sqrt((2 * vsv**2 + vsh**2) / 3)

vs_2d = vs_voigt.reshape(theta_grid.shape)
row_mean = np.nanmean(vs_2d, axis=1, keepdims=True)
dvs_2d = vs_2d - row_mean

print(f"Vs range:  {np.nanmin(vs_2d):.0f} to {np.nanmax(vs_2d):.0f} m/s")
print(f"dVs range: {np.nanmin(dvs_2d):.0f} to {np.nanmax(dvs_2d):.0f} m/s")
# -

# Visualisation
# -------------
#
# We plot the cross-section on a polar projection, which naturally
# represents the curvature of the Earth. The radial axis spans from the
# core-mantle boundary to the surface, and the angular axis shows the
# arc position. We use a diverging colour scale centred at zero so that
# blue indicates faster-than-average (colder) material and red indicates
# slower-than-average (hotter) material.
#
# The slow anomaly in the lower mantle beneath Africa is the African
# LLSVP, one of two continent-sized thermochemical structures that
# dominate the pattern of long-wavelength mantle heterogeneity.

# + tags=["active-ipynb"]
# %matplotlib inline
# -

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
#
# half_span_deg = arc_deg / 2.0
#
# fig, ax = plt.subplots(
#     figsize=(5, 5), subplot_kw={"projection": "polar"}, dpi=120
# )
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
#
# vmax = np.nanpercentile(np.abs(dvs_2d), 98)
# norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
#
# pcm = ax.pcolormesh(
#     theta_grid, r_grid / 1e3, dvs_2d,
#     cmap="RdBu", norm=norm, shading="auto",
# )
#
# ax.set_thetamin(-half_span_deg)
# ax.set_thetamax(half_span_deg)
# ax.set_rorigin(0)
# ax.set_rlim(R_cmb / 1e3, R_earth / 1e3)
# ax.set_yticks([])
# ax.grid(False)
#
# ax.set_xticks([
#     np.radians(-half_span_deg),
#     0,
#     np.radians(half_span_deg),
# ])
# ax.set_xticklabels([
#     f"{abs(lat_A):.1f}\u00b0{'S' if lat_A < 0 else 'N' if lat_A > 0 else ''}",
#     "Mid",
#     f"{abs(lat_B):.1f}\u00b0{'S' if lat_B < 0 else 'N' if lat_B > 0 else ''}",
# ], fontsize=8)
#
# ax.set_title("REVEAL Cross-Section\nMid-Atlantic to Afar", fontsize=11, pad=12)
#
# cax = fig.add_axes([0.3, 0.22, 0.4, 0.02])
# cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
# cbar.set_label(r"$\Delta V_s$ (m/s)", fontsize=9)
# cbar.ax.tick_params(labelsize=7)
#
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example showed how to:
#
# - Load a tomography model with ``gdrift.SeismicModel("REVEAL")``
# - Build a great-circle cross-section grid with
#   ``gdrift.great_circle_cross_section``
# - Compute isotropic Vs from anisotropic components using the Voigt
#   average
# - Remove the depth-by-depth lateral mean to obtain perturbations from
#   absolute velocities
# - Plot the result on a polar wedge projection
#
# The same workflow applies to any global model. For regional models, a
# depth-slice approach (as in the `Seismic Tomography
# <../tomography_models/demo.ipynb>`_ example) is more appropriate.
