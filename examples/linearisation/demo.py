# Linearising Thermodynamic Lookup Tables
# ========================================
#
# This example demonstrates how to regularise (linearise) thermodynamic
# property tables so that sharp discontinuities at mantle phase transitions
# are smoothed out. The regularised tables are essential for two key
# applications in mantle convection modelling:
#
# 1. **Inverse velocity-to-temperature conversion** -- phase-transition jumps
#    make the $V_s(T)$ relationship multi-valued at certain depths, so a
#    unique inverse does not exist without smoothing.
# 2. **Gradient-based numerical methods** -- finite-element solvers require
#    smooth fields; abrupt jumps in material properties cause convergence
#    problems.
#
# Background
# ----------
#
# Thermodynamic lookup tables such as those from Stixrude & Lithgow-Bertelloni
# store material properties (density $\rho$, shear-wave speed $V_s$,
# compressional-wave speed $V_p$, etc.) on a regular
# (depth $\times$ temperature) grid. At phase-transition boundaries
# (~410 km and ~660 km in the mantle), these properties exhibit sharp jumps
# because the stable mineral assemblage changes abruptly.
#
# The regularisation algorithm works as follows:
#
# 1. Compute the temperature gradient $\partial V / \partial T$ of each
#    property across the 2D table.
# 2. Identify *irregular* gradient values that fall outside an acceptable
#    range (e.g. large positive jumps in $V_s$ that correspond to phase
#    transitions).
# 3. Replace irregular gradients with inverse-distance-weighted interpolation
#    from nearby regular points using a KD-tree.
# 4. Integrate the smoothed gradient back along the temperature axis to
#    reconstruct the property field.
# 5. Anchor the reconstructed field to a reference temperature profile so
#    that properties along the geotherm are preserved exactly.
#
# This example
# ------------
#
# We load the SLB_21 pyrolite (CFMAS) thermodynamic model, define a simple
# reference temperature profile, and apply `regularise_thermodynamic_table`
# to produce a smoothed model. We then compare the original and regularised
# $V_s$, $V_p$, and $\rho$ at four key depths spanning the major phase
# transitions.

import numpy as np
import gdrift

# Loading the Thermodynamic Model
# --------------------------------
#
# We load the SLB_21 pyrolite model on a custom grid spanning 300--4000 K
# in temperature and 0--2890 km in depth. The explicit grid is needed so
# that both the original and regularised tables share the same axes.

# +
slb_pyrolite = gdrift.ThermodynamicModel(
    "SLB_21", "pyroliteCFMAS",
    temps=np.linspace(300, 4000),
    depths=np.linspace(0, 2890e3),
)
print(f"Table shape: {len(slb_pyrolite.get_depths())} depths x "
      f"{len(slb_pyrolite.get_temperatures())} temperatures")
# -

# Defining a Reference Temperature Profile
# ------------------------------------------
#
# The regularisation anchors the smoothed tables to a reference geotherm.
# Along this profile, the regularised properties match the original model
# exactly. Away from the anchor, properties vary smoothly without
# phase-transition jumps.
#
# We use a simple four-point spline that roughly follows an adiabatic
# mantle geotherm.

# +
temperature_profile = gdrift.SplineProfile(
    depth=np.asarray([0., 500e3, 2700e3, 3000e3]),
    value=np.asarray([300, 1000, 3000, 4000]),
)

anchor_depths = np.array([0, 500, 1500, 2700]) * 1e3
for d in anchor_depths:
    print(f"  T at {d/1e3:.0f} km: {temperature_profile.at_depth(d):.0f} K")
# -

# Regularising the Model
# -----------------------
#
# The `regularise_thermodynamic_table` function takes the original model,
# the reference temperature profile, and an optional `regular_range` dict
# that specifies acceptable bounds for $\partial V / \partial T$. Gradient
# values outside this range are flagged as irregular and smoothed.
#
# The default range prohibits any positive gradient (i.e. no phase-transition
# jumps). Here we additionally cap the magnitude of the negative $V_s$
# gradient at $-1.5$, which further suppresses residual kinks.

# +
regular_slb = gdrift.regularise_thermodynamic_table(
    slb_pyrolite,
    temperature_profile,
    regular_range={
        "v_s": (-1.5, 0.0),
        "v_p": (-np.inf, 0.0),
        "rho": (-np.inf, 0.0),
    },
)
print("Regularised model created successfully.")
# -

# Extracting Original and Regularised Tables
# --------------------------------------------
#
# We extract the 2D property arrays from both models for comparison.
# Each array has shape `(n_depths, n_temperatures)`.

# +
Vs_original = slb_pyrolite.compute_swave_speed().get_vals()
Vp_original = slb_pyrolite.compute_pwave_speed().get_vals()
rho_original = slb_pyrolite._tables["rho"].get_vals()

Vs_regularised = regular_slb.compute_swave_speed().get_vals()
Vp_regularised = regular_slb.compute_pwave_speed().get_vals()
rho_regularised = regular_slb._tables["rho"].get_vals()

temperatures = slb_pyrolite.get_temperatures()
all_depths = slb_pyrolite.get_depths()
# -

# Comparing at Key Depths
# ------------------------
#
# We compare the original and regularised tables at four depths that span
# the major upper-mantle phase transitions (olivine to wadsleyite at
# ~410 km, ringwoodite to bridgmanite + ferropericlase at ~660 km) and
# the mid/lower mantle (1000 km, 2000 km).

# +
comparison_depths = np.array([410, 660, 1000, 2000]) * 1e3
depth_indices = [np.abs(d - all_depths).argmin() for d in comparison_depths]

for idx in depth_indices:
    d_km = all_depths[idx] / 1e3
    anchor_T = temperature_profile.at_depth(all_depths[idx])
    t_idx = np.abs(temperatures - anchor_T).argmin()
    print(f"\nDepth {d_km:.0f} km (anchor T = {anchor_T:.0f} K):")
    print(f"  Vs  original={Vs_original[idx, t_idx]:.1f}  regularised={Vs_regularised[idx, t_idx]:.1f} m/s")
    print(f"  Vp  original={Vp_original[idx, t_idx]:.1f}  regularised={Vp_regularised[idx, t_idx]:.1f} m/s")
    print(f"  rho original={rho_original[idx, t_idx]:.1f}  regularised={rho_regularised[idx, t_idx]:.1f} kg/m^3")
# -

# Visualisation
# -------------
#
# For each of the four depths, we plot the original and regularised
# profiles of $V_s$, $V_p$, and $\rho$ as a function of temperature.
# The green dashed line marks the anchor temperature from the reference
# profile. Notice how the sharp phase-transition jumps in the original
# tables (blue) are smoothed in the regularised version (red), while the
# two curves agree at the anchor temperature.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# fig, axs = plt.subplots(len(depth_indices), 3, figsize=(15, 10),
#                          constrained_layout=True)
#
# column_labels = [r"$V_s$ [m/s]", r"$V_p$ [m/s]", r"$\rho$ [kg/m$^3$]"]
#
# for i, idx in enumerate(depth_indices):
#     depth_km = all_depths[idx] / 1e3
#     anchor_T = temperature_profile.at_depth(all_depths[idx])
#
#     for j, (orig, reg, ylabel) in enumerate(zip(
#         [Vs_original, Vp_original, rho_original],
#         [Vs_regularised, Vp_regularised, rho_regularised],
#         column_labels,
#     )):
#         ax = axs[i, j]
#         ax.plot(temperatures, orig[idx, :], color="blue", label="Original")
#         ax.plot(temperatures, reg[idx, :], color="red", label="Regularised")
#         ax.axvline(x=anchor_T, color="green", linestyle="--",
#                    label="Anchor T")
#         ax.set_title(f"{ylabel} at {depth_km:.0f} km")
#         ax.set_xlabel("Temperature [K]")
#         ax.grid(alpha=0.3)
#         if i == 0:
#             ax.legend(fontsize=8)
#
# fig.suptitle("Original vs Regularised Thermodynamic Properties", fontsize=14)
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated how to:
#
# - Load a thermodynamic model with a custom depth/temperature grid
# - Define a reference temperature profile using `gdrift.SplineProfile`
# - Regularise the model with `gdrift.regularise_thermodynamic_table()`
# - Compare original and smoothed property tables at phase-transition depths
#
# The regularised model can be used as a drop-in replacement for the
# original `ThermodynamicModel` in any downstream workflow (e.g. anelastic
# corrections, velocity-to-temperature inversion).
