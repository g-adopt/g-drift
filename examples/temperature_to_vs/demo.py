# Converting Seismic Velocity to Temperature
# ============================================
#
# This example demonstrates how to build a complete pipeline for converting
# seismic shear-wave velocities from a 3D tomography model into mantle
# temperature. The pipeline combines four components:
#
# 1. **Thermodynamic lookup tables** (SLB_21) -- the relationship between
#    temperature, depth, and elastic seismic velocity.
# 2. **Table regularisation** -- smoothing discontinuities at mantle phase
#    transitions (410 km, 660 km) to enable a unique inverse mapping.
# 3. **Anelastic corrections** -- shifting from elastic (infinite-frequency)
#    to seismic-frequency ($\sim 1\,\mathrm{Hz}$) velocities.
# 4. **3D seismic tomography** (REVEAL) -- observed shear-wave velocities
#    as input to the conversion.
#
# Background
# ----------
#
# Elastic velocities from mineral physics differ from seismic observations
# due to anelastic attenuation. The velocity--temperature relationship
# $V_s(T, z)$ is also multi-valued near phase transitions (~410 km,
# ~660 km). A robust $V_s \to T$ inversion therefore requires:
#
# - Regularisation of the thermodynamic table to remove phase-transition
#   jumps.
# - Anelastic corrections to match seismic observation frequencies.
#
# This example
# ------------
#
# We use the SLB_21 pyrolite (CFMAS) thermodynamic model -- the
# Stixrude & Lithgow-Bertelloni (2021) database with a five-component
# (CaO--FeO--MgO--Al$_2$O$_3$--SiO$_2$) composition. A realistic
# reference temperature profile from a mantle convection simulation
# (including upper and lower thermal boundary layers) anchors the
# regularisation. The Cammarano Q3 attenuation profile provides the
# anelastic correction. Finally, we query the REVEAL global
# full-waveform inversion model (Thrastarson et al. 2024) at 200 km
# depth and convert its isotropic shear-wave speed to temperature.

from pathlib import Path

import numpy as np
import gdrift

_demo_dir = Path(__file__).parent

# Loading the SLB_21 Thermodynamic Model
# ----------------------------------------
#
# We load the SLB_21 pyrolite model in the CFMAS
# (CaO--FeO--MgO--Al$_2$O$_3$--SiO$_2$) system. The model provides
# 2D lookup tables of material properties (density, elastic moduli,
# velocities) on a depth $\times$ temperature grid.

# +
slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")

print(f"Available tables: {slb21.available_tables()}")
print(f"Depth range: {slb21.get_depths().min()/1e3:.0f}"
      f" - {slb21.get_depths().max()/1e3:.0f} km")
print(f"Temperature range: {slb21.get_temperatures().min():.0f}"
      f" - {slb21.get_temperatures().max():.0f} K")
# -

# Building a Reference Temperature Profile
# ------------------------------------------
#
# The regularisation algorithm anchors the smoothed thermodynamic tables
# to a reference temperature profile. A pure adiabat is inadequate for
# this purpose: it starts at the potential temperature (~1600 K) and
# misses the thermal boundary layers at the top and bottom of the mantle.
# In reality, the temperature rises from ~300 K at the surface through
# the lithospheric boundary layer to the adiabat, and similarly increases
# from the adiabat to ~4200 K at the CMB through the D$''$ layer.
#
# Here we use an azimuthally-averaged temperature profile from a mantle
# convection simulation (Terra code), which captures both boundary layers
# and provides a physically realistic anchor.

# +
terra_data = np.loadtxt(
    _demo_dir.parent / "TerraMT512vs.dat", unpack=False, usecols=(0, 1))

temperature_profile = gdrift.SplineProfile(
    depth=terra_data[:, 0] * 1e3,
    value=terra_data[:, 1],
    name="Terra average temperature",
    extrapolate=True,
)

for d in [0, 100, 200, 660, 2000, 2890]:
    print(f"  T at {d:>4d} km: {temperature_profile.at_depth(d * 1e3):7.0f} K")
# -

# Regularising the Thermodynamic Table
# --------------------------------------
#
# Phase transitions at ~410 km and ~660 km create sharp jumps in $V_s(T)$
# that make the inverse mapping multi-valued. The regularisation replaces
# these jumps with smooth gradients while preserving the values along the
# reference profile exactly. The ``regular_range`` parameter specifies the
# acceptable bounds for $\partial V / \partial T$: we allow only negative
# gradients (velocity decreases with temperature), capping $V_s$ gradients
# at $-1.5$ to suppress residual kinks.

# +
regular_slb21 = gdrift.regularise_thermodynamic_table(
    slb21,
    temperature_profile,
    regular_range={
        "v_s": (-1.5, 0.0),
        "v_p": (-np.inf, 0.0),
        "rho": (-np.inf, 0.0),
    },
)
print("Regularised model created.")
# -

# Applying Anelastic Corrections
# --------------------------------
#
# Anelastic attenuation reduces seismic velocities relative to the elastic
# (infinite-frequency) values from the thermodynamic model. The correction
# depends on the quality factor $Q(T, z)$, parameterised here using the
# Cammarano Q3 profile:
#
# $$V_{\mathrm{anelastic}} = V_{\mathrm{elastic}}
#    \left(1 - \frac{1}{2\tan(a\pi/2)\,Q}\right)$$
#
# where $a \approx 0.2$ is the frequency exponent.

# +
anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile("Q3")
corrected_slb21 = gdrift.apply_anelastic_correction(regular_slb21, anelastic)
print("Anelastic correction applied.")
# -

# Comparing Elastic and Corrected Velocities
# --------------------------------------------
#
# At each depth, the anelastic correction reduces $V_s$ by a temperature-
# dependent amount. The effect is strongest at high temperatures where
# attenuation (low $Q$) is most significant. We compare the elastic and
# corrected velocity profiles at four key depths.

# +
comparison_depths = np.array([200, 410, 660, 1000]) * 1e3
test_temperatures = np.linspace(1000, 3500, 50)

Vs_elastic = np.array([
    slb21.temperature_to_vs(test_temperatures, d) for d in comparison_depths
])
Vs_corrected = np.array([
    corrected_slb21.temperature_to_vs(test_temperatures, d)
    for d in comparison_depths
])

for i, d in enumerate(comparison_depths):
    reduction = (Vs_elastic[i] - Vs_corrected[i]) / Vs_elastic[i] * 100
    print(f"Depth {d/1e3:.0f} km: max Vs reduction = {reduction.max():.1f}%")
# -

# Loading the REVEAL Tomography Model
# -------------------------------------
#
# REVEAL (Thrastarson et al. 2024) is a global full-waveform inversion
# model providing absolute velocities ($V_{SV}$, $V_{SH}$, $V_{PV}$)
# and density. We extract a depth slice at 200 km, which samples the
# lithosphere--asthenosphere system where thermal structure produces
# strong velocity variations.

# +
seismic_model = gdrift.SeismicModel("REVEAL")

slice_depth = 200e3
lats = np.linspace(-85, 85, 35)
lons = np.linspace(-180, 175, 72)
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
depth_grid = np.full_like(lat_grid, slice_depth)

coordinates = gdrift.geodetic_to_cartesian(
    lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel())

reveal_data = seismic_model.at(["vsh", "vsv"], coordinates)
vsh = reveal_data[:, 0]
vsv = reveal_data[:, 1]

print(f"Queried {len(vsh)} points at {slice_depth/1e3:.0f} km depth")
# -

# Computing Isotropic Shear-Wave Speed
# --------------------------------------
#
# REVEAL provides anisotropic velocities ($V_{SH}$ and $V_{SV}$). We
# compute the isotropic Voigt-average shear-wave speed:
#
# $$V_S = \sqrt{\frac{2\,V_{SH}^2 + V_{SV}^2}{3}}$$

# +
vs_isotropic = np.sqrt((2 * vsh**2 + vsv**2) / 3)

print(f"Vs range at {slice_depth/1e3:.0f} km: "
      f"{np.nanmin(vs_isotropic):.0f} - {np.nanmax(vs_isotropic):.0f} m/s")
# -

# Converting Velocity to Temperature
# ------------------------------------
#
# Using the regularised and anelastically-corrected model, we invert
# $V_s \to T$ at each grid point. The method uses bounded scalar
# minimisation of $|V_s(T, z) - V_s^{\mathrm{obs}}|^2$ at fixed depth.

# +
valid = np.isfinite(vs_isotropic)
converted_temperature = np.full_like(vs_isotropic, np.nan)
converted_temperature[valid] = corrected_slb21.vs_to_temperature(
    vs_isotropic[valid], depth_grid.ravel()[valid])

print(f"Temperature range at {slice_depth/1e3:.0f} km: "
      f"{np.nanmin(converted_temperature):.0f}"
      f" - {np.nanmax(converted_temperature):.0f} K")
# -

# Visualisation
# -------------
#
# We plot the elastic-versus-corrected $V_s(T)$ at four depths, the
# REVEAL isotropic $V_s$ at 200 km, and the converted temperature field.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(16, 12), constrained_layout=True)
# gs = fig.add_gridspec(2, 2)
#
# # --- Panel 1: Elastic vs Corrected Vs(T) at four depths ---
# ax1 = fig.add_subplot(gs[0, 0])
# colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
# for i, d in enumerate(comparison_depths):
#     ax1.plot(test_temperatures, Vs_elastic[i], color=colors[i],
#              linestyle="--", alpha=0.6)
#     ax1.plot(test_temperatures, Vs_corrected[i], color=colors[i],
#              label=f"{d/1e3:.0f} km")
# ax1.set_xlabel("Temperature [K]")
# ax1.set_ylabel(r"$V_s$ [m/s]")
# ax1.set_title(r"Elastic (dashed) vs Corrected $V_s$")
# ax1.legend()
# ax1.grid(alpha=0.3)
#
# # --- Panel 2: Velocity reduction ---
# ax2 = fig.add_subplot(gs[0, 1])
# for i, d in enumerate(comparison_depths):
#     reduction = (Vs_elastic[i] - Vs_corrected[i]) / Vs_elastic[i] * 100
#     ax2.plot(test_temperatures, reduction, color=colors[i],
#              label=f"{d/1e3:.0f} km")
# ax2.set_xlabel("Temperature [K]")
# ax2.set_ylabel("Velocity reduction [%]")
# ax2.set_title("Anelastic velocity reduction")
# ax2.legend()
# ax2.grid(alpha=0.3)
#
# # --- Panel 3: REVEAL Vs at 200 km ---
# ax3 = fig.add_subplot(gs[1, 0])
# vs_map = vs_isotropic.reshape(lat_grid.shape)
# img3 = ax3.pcolormesh(lon_grid, lat_grid, vs_map,
#                        cmap="seismic_r", shading="auto")
# fig.colorbar(img3, ax=ax3, label=r"$V_s$ [m/s]")
# ax3.set_xlabel("Longitude")
# ax3.set_ylabel("Latitude")
# ax3.set_title(f"REVEAL isotropic Vs at {slice_depth/1e3:.0f} km")
#
# # --- Panel 4: Converted temperature at 200 km ---
# ax4 = fig.add_subplot(gs[1, 1])
# temp_map = converted_temperature.reshape(lat_grid.shape)
# img4 = ax4.pcolormesh(lon_grid, lat_grid, temp_map,
#                        cmap="hot", shading="auto")
# fig.colorbar(img4, ax=ax4, label="Temperature [K]")
# ax4.set_xlabel("Longitude")
# ax4.set_ylabel("Latitude")
# ax4.set_title(f"Converted temperature at {slice_depth/1e3:.0f} km")
#
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated the full velocity-to-temperature conversion
# pipeline:
#
# - Loaded the SLB_21 pyrolite (CFMAS) thermodynamic model.
# - Used a simulation-derived reference temperature profile that captures
#   the upper and lower thermal boundary layers.
# - Regularised the table to smooth phase-transition discontinuities.
# - Applied Cammarano Q3 anelastic corrections.
# - Loaded the REVEAL full-waveform inversion model.
# - Computed isotropic $V_s$ from anisotropic components via Voigt average.
# - Inverted $V_s \to T$ at 200 km depth across the globe.
#
# The same pipeline can be applied at any depth or with different
# thermodynamic models, compositions, and attenuation parameterisations.
