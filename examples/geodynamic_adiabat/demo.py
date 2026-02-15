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
# We compute and compare adiabatic profiles using two versions of the
# Stixrude & Lithgow-Bertelloni thermodynamic database: SLB_21
# (pyroliteCFMAS) and SLB_24 (pyroliteCFMS), starting from a surface
# potential temperature of 1600 K. The SLB_24 database is available with
# the CFMS chemical system (CaO-FeO-MgO-SiO2), while SLB_21 includes
# the fuller CFMAS system (with Al2O3).

import numpy as np
import gdrift

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

# Loading Thermodynamic Models
# -----------------------------
#
# We load two pyrolite models from the Stixrude & Lithgow-Bertelloni
# thermodynamic databases: the 2021 version (CFMAS chemical system) and
# the 2024 version (CFMS chemical system).

# +
slb21 = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
slb24 = gdrift.ThermodynamicModel("SLB_24", "pyroliteCFMS")
# -

# Computing Adiabatic Profiles
# -----------------------------
#
# The `compute_adiabat` function integrates the adiabatic gradient ODE
# from a surface potential temperature $T_0$ and evaluates material
# properties along the resulting temperature profile.

# +
T0 = 1600  # surface potential temperature [K]

adiabat_21 = gdrift.compute_adiabat(slb21, T0=T0, gravity_profile=gravity_profile)
adiabat_24 = gdrift.compute_adiabat(slb24, T0=T0, gravity_profile=gravity_profile)
# -

# Comparing Results
# -----------------
#
# The two databases yield slightly different adiabats due to updated
# thermodynamic parameters and different chemical systems. We compare
# CMB temperatures, dissipation numbers, and surface properties.

# +
for label, adiabat in [("SLB_21 (CFMAS)", adiabat_21), ("SLB_24 (CFMS)", adiabat_24)]:
    print(f"\n{label}:")
    print(f"  CMB temperature:   {adiabat['temperature'][-1]:.0f} K")
    print(f"  Dissipation number: {adiabat['Di']:.3f}")
    print(f"  Surface alpha:     {adiabat['alpha'][0]:.3e} 1/K")
    print(f"  Surface Cp_SI:     {adiabat['Cp_SI'][0]:.1f} J/kg/K")
    print(f"  Surface rho:       {adiabat['rho'][0]:.1f} kg/m^3")
# -

# Visualisation
# -------------
#
# We plot key profiles side by side to compare the two thermodynamic models.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharey=True)
# depths_km = adiabat_21["depths"] / 1e3
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
#     ax.plot(adiabat_21[key], depths_km, label="SLB_21 (CFMAS)")
#     ax.plot(adiabat_24[key], depths_km, label="SLB_24 (CFMS)", linestyle="--")
#     ax.set_xlabel(xlabel, fontsize=10)
#     ax.grid(alpha=0.3)
#     ax.invert_yaxis()
#     ax.legend(fontsize=8)
#
# axes[0, 0].set_ylabel("Depth [km]", fontsize=11)
# axes[1, 0].set_ylabel("Depth [km]", fontsize=11)
#
# fig.suptitle(
#     f"Adiabatic Profiles ($T_0$ = {T0} K)\n"
#     f"Di(SLB_21) = {adiabat_21['Di']:.3f}, "
#     f"Di(SLB_24) = {adiabat_24['Di']:.3f}",
#     fontsize=13,
# )
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated how to:
# - Build a gravity profile from PREM using `gdrift.prem_gravity_profile()`
# - Compute adiabatic temperature profiles using `gdrift.compute_adiabat()`
# - Compare adiabats from different thermodynamic databases (SLB_21 vs SLB_24)
# - Compute the dissipation number $Di$ for each model
#
# The adiabatic profiles computed here serve as the reference state for
# mantle convection simulations. The dissipation number indicates that
# compressibility effects are significant for Earth's mantle.
