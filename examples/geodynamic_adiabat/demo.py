# Computing Geodynamic Adiabatic Profiles
# ========================================
#
# This example demonstrates how to compute 1D adiabatic temperature profiles
# through the mantle using self-consistent thermodynamic models. Adiabatic
# profiles define the reference thermal state for mantle convection and are
# essential inputs for computing density anomalies, seismic velocity
# perturbations, and the dissipation number.
#
# Background
# ----------
#
# In a well-mixed convecting mantle, temperature increases with depth along
# an adiabat. The adiabatic gradient is governed by the ODE:
#
# $$\frac{dT}{dz} = \frac{\alpha \, T \, g}{C_p^{\mathrm{SI}}}$$
#
# where $\alpha$ is thermal expansivity, $T$ is temperature, $g$ is
# gravitational acceleration, and $C_p^{\mathrm{SI}} = C_p / (\rho V)$ is
# the specific heat capacity in SI units (J kg$^{-1}$ K$^{-1}$). The
# thermodynamic properties ($\alpha$, $\rho$, $C_p$, $V$) are themselves
# functions of temperature and depth, making the ODE nonlinear.
#
# Phase transitions at ~410 km and ~660 km depth introduce sharp
# discontinuities in material properties. While the ODE integration
# produces a smooth temperature profile (the adiabat itself is continuous),
# the evaluated properties inherit these jumps from the underlying
# thermodynamic tables. We apply a Savitzky-Golay filter to the property
# profiles to obtain smooth reference profiles suitable for mantle
# convection simulations.
#
# The Dissipation Number
# ----------------------
#
# The dissipation number $Di$ quantifies the importance of adiabatic
# heating and viscous dissipation relative to convective heat transport:
#
# $$Di = \frac{\alpha_s \, g_s \, D}{C_{p,s}^{\mathrm{SI}}}$$
#
# where the subscript $s$ denotes surface values and $D = R_{\mathrm{earth}}
# - R_{\mathrm{cmb}}$ is the mantle thickness. For $Di \ll 1$ the
# Boussinesq approximation is valid; Earth's mantle has $Di \approx 0.5$--$0.7$,
# indicating that compressibility effects are significant and the extended
# Boussinesq or fully compressible formulations are needed.
#
# This example
# ------------
#
# We compute an adiabatic profile using the Stixrude & Lithgow-Bertelloni
# 2021 thermodynamic database with a pyrolite composition in the FMS
# (FeO-MgO-SiO$_2$) chemical system, starting from a surface potential
# temperature of 1600 K. The raw profiles are then smoothed with a
# Savitzky-Golay filter to remove phase-transition spikes, and we compare
# the raw and smoothed results.

import gdrift
import numpy as np
from scipy.signal import savgol_filter

# Gravity from PREM
# -----------------
#
# We first compute a gravity profile from PREM's density structure. This
# profile will be used by the adiabat integration.

# +
gravity_profile = gdrift.prem_gravity_profile()
print(f"Surface gravity: {gravity_profile.at_depth(0):.3f} m/s^2")
print(f"CMB gravity:     {gravity_profile.at_depth(2890e3):.3f} m/s^2")
# -

# Loading the Thermodynamic Model
# --------------------------------
#
# We load the SLB_21 pyrolite model in the FMS chemical system
# (FeO-MgO-SiO$_2$). This simplified system captures the dominant
# mantle minerals and produces adiabatic profiles in close agreement
# with published reference models.

# +
slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteFMS")
# -

# Computing the Adiabatic Profile
# --------------------------------
#
# The `compute_adiabat` function integrates the adiabatic gradient ODE
# from a surface potential temperature $T_0$ and evaluates material
# properties along the resulting temperature profile.

# +
T0 = 1600  # surface potential temperature [K]

adiabat = gdrift.compute_adiabat(slb21, T0=T0, gravity_profile=gravity_profile)
# -

# Smoothing Phase-Transition Discontinuities
# --------------------------------------------
#
# The raw property profiles contain sharp jumps at phase transitions
# (notably the olivine-wadsleyite transition at ~410 km and the
# post-spinel transition at ~660 km). These discontinuities are physical
# but problematic for numerical simulations that require smooth reference
# profiles. We apply a Savitzky-Golay filter (window = 21 points
# $\approx$ 240 km, cubic polynomial) to remove the spikes while
# preserving the large-scale depth dependence.

# +
smooth_keys = ["rho", "alpha", "Cp", "V", "Cv", "beta", "gamma"]

adiabat_smooth = dict(adiabat)  # shallow copy
for key in smooth_keys:
    adiabat_smooth[key] = savgol_filter(adiabat[key], window_length=21, polyorder=3)

# Recompute derived SI heat capacities from smoothed fields
adiabat_smooth["Cp_SI"] = adiabat_smooth["Cp"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])
adiabat_smooth["Cv_SI"] = adiabat_smooth["Cv"] / (adiabat_smooth["rho"] * adiabat_smooth["V"])
# -

# Results
# -------
#
# We print key values from both the raw and smoothed profiles. The
# temperature and dissipation number are unchanged by smoothing (the
# temperature profile is already smooth from the ODE integration, and
# $Di$ depends only on surface values).

# +
print(f"\nSLB_21 pyrolite FMS (T0 = {T0} K):")
print(f"  CMB temperature:    {adiabat['temperature'][-1]:.0f} K")
print(f"  Dissipation number: {adiabat['Di']:.3f}")
print(f"\n  Surface properties (raw / smoothed):")
print(f"    alpha:  {adiabat['alpha'][0]:.3e} / {adiabat_smooth['alpha'][0]:.3e} 1/K")
print(f"    Cp_SI:  {adiabat['Cp_SI'][0]:.1f} / {adiabat_smooth['Cp_SI'][0]:.1f} J/kg/K")
print(f"    rho:    {adiabat['rho'][0]:.1f} / {adiabat_smooth['rho'][0]:.1f} kg/m^3")
# -

# Visualisation
# -------------
#
# We plot the raw profiles (thin, translucent) overlaid with the smoothed
# profiles (solid) to highlight the effect of the Savitzky-Golay filter
# on phase-transition discontinuities.

# + tags=["active-ipynb"]
# %matplotlib inline
# -

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharey=True)
# depths_km = adiabat["depths"] / 1e3
#
# plot_specs = [
#     ("temperature", "Temperature [K]"),
#     ("rho", "Density [kg/m$^3$]"),
#     ("alpha", "Thermal Expansivity [1/K]"),
#     ("Cp_SI", "Cp [J/kg/K]"),
#     ("gravity", "Gravity [m/s$^2$]"),
#     ("Cv_SI", "Cv [J/kg/K]"),
# ]
#
# for ax, (key, xlabel) in zip(axes.flat, plot_specs):
#     ax.plot(adiabat[key], depths_km, color="0.7", linewidth=0.8, label="Raw")
#     ax.plot(adiabat_smooth[key], depths_km, linewidth=1.5, label="Smoothed")
#     ax.set_xlabel(xlabel, fontsize=10)
#     ax.grid(alpha=0.3)
#     ax.invert_yaxis()
#     ax.legend(fontsize=8)
#
# axes[0, 0].set_ylabel("Depth [km]", fontsize=11)
# axes[1, 0].set_ylabel("Depth [km]", fontsize=11)
#
# fig.suptitle(
#     f"Adiabatic Profile — SLB_21 pyrolite FMS ($T_0$ = {T0} K)\n"
#     f"Di = {adiabat['Di']:.3f}, CMB T = {adiabat['temperature'][-1]:.0f} K",
#     fontsize=13,
# )
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated how to:
#
# - Build a gravity profile from PREM using `gdrift.prem_gravity_profile()`
# - Compute an adiabatic temperature profile using `gdrift.compute_adiabat()`
# - Smooth phase-transition discontinuities with a Savitzky-Golay filter
# - Compute the dissipation number $Di$
#
# The adiabatic profiles computed here serve as the reference state for
# mantle convection simulations. The smoothed profiles remove
# phase-transition spikes while preserving the large-scale depth
# dependence of material properties.
