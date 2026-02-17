# Anelastic Corrections for Seismic Velocities
# ==============================================
#
# This example demonstrates how to apply frequency-dependent anelastic
# corrections to elastic seismic velocities computed from mineral physics
# lookup tables. We cover all eight available Q-profiles: six from the
# Cammarano et al. (2003) parameterization (Q1--Q6) and two from the
# Goes et al. (2000) parameterization (Q4, Q6).
#
# Background
# ----------
#
# Seismic velocities measured at seismic frequencies (~1 Hz) are lower
# than the elastic (infinite-frequency) velocities predicted by mineral
# physics. This difference arises because mantle rocks are not perfectly
# elastic: they dissipate energy through anelastic processes such as
# grain-boundary sliding and dislocation creep.
#
# The quality factor $Q$ quantifies attenuation. The velocity correction
# in the low-attenuation limit is:
#
# $$\frac{V_{\text{anel}}}{V_{\text{el}}} = 1 - \frac{1}{2}
#   \cot\!\left(\frac{a\pi}{2}\right) Q^{-1}$$
#
# where $a$ is the frequency exponent. Two parameterizations for
# $Q(\text{depth}, T)$ are implemented in gdrift:
#
# **Cammarano et al. (2003)**:
# $Q = B \, \omega^{a} \exp\!\left(\frac{a \, g \, T_s}{T}\right)$
# where $B$ relates to grain size, $g$ is a dimensionless activation
# energy, and $T_s$ is the solidus temperature.
#
# **Goes et al. (2000)**:
# $Q = Q_0 \, \omega^{a} \exp\!\left(\frac{a \, \xi \, T_s}{T}\right)$
# where $Q_0$ is a reference quality factor and $\xi$ is the activation
# parameter.
#
# For P-wave corrections, the quality factor is computed from $Q_\mu$
# (shear) and $Q_\kappa$ (bulk) following Anderson & Hart (1978).
#
# This example
# ------------
#
# We demonstrate how to:
# 1. Load an elastic thermodynamic model (SLB_21 pyroliteCFMAS)
# 2. Create anelasticity models for all 8 Q-profiles using factory methods
# 3. Apply anelastic corrections to obtain seismic-frequency velocities
# 4. Compare velocity reductions across all models

import numpy as np
import gdrift

# Loading the Elastic Thermodynamic Model
# ----------------------------------------
#
# We use the SLB_21 pyroliteCFMAS thermodynamic table, which provides
# elastic velocities (Vs, Vp) and other properties as a function of
# depth and temperature.

# +
slb_pyrolite = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
print(f"Available tables: {slb_pyrolite.available_tables()}")
print(f"Depth range: {slb_pyrolite.get_depths().min() / 1e3:.0f} -- "
      f"{slb_pyrolite.get_depths().max() / 1e3:.0f} km")
print(f"Temperature range: {slb_pyrolite.get_temperatures().min():.0f} -- "
      f"{slb_pyrolite.get_temperatures().max():.0f} K")
# -

# Cammarano Q-Profiles (Q1--Q6)
# ------------------------------
#
# The Cammarano parameterization provides six Q-profiles that represent
# different assumptions about grain size and activation energy in the
# mantle. Q1--Q3 use Cammarano et al. (2003) activation energies, while
# Q4--Q6 use Stixrude & Lithgow-Bertelloni values. All switch parameters
# at the 660 km discontinuity between upper and lower mantle.
#
# We use the `from_q_profile()` factory method to create each model,
# then apply the anelastic correction with `apply_anelastic_correction()`.

# +
cammarano_q_names = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
cammarano_models = {}

for qname in cammarano_q_names:
    anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile(qname)
    corrected = gdrift.apply_anelastic_correction(slb_pyrolite, anelastic)
    key = f"Cammarano_{qname}"
    cammarano_models[key] = corrected
    print(f"  {key}: created")
# -

# Goes Q-Profiles (Q4, Q6)
# -------------------------
#
# The Goes et al. (2000) parameterization uses a different functional
# form with parameters $Q_0$ and $\xi$. Two profiles are available: Q4
# (standard) and Q6 (alternative). The frequency exponent is $a = 0.15$
# compared to $a = 0.2$ for Cammarano.

# +
goes_q_names = ["Q4", "Q6"]
goes_models = {}

for qname in goes_q_names:
    anelastic = gdrift.GoesAnelasticityModel.from_q_profile(qname)
    corrected = gdrift.apply_anelastic_correction(slb_pyrolite, anelastic)
    key = f"Goes_{qname}"
    goes_models[key] = corrected
    print(f"  {key}: created")
# -

# Combine all models for comparison:

# +
all_models = {}
all_models.update(cammarano_models)
all_models.update(goes_models)
print(f"Total anelastic models: {len(all_models)}")
# -

# Comparing Velocity Reductions
# ------------------------------
#
# We query elastic and anelastic shear-wave velocities at a fixed depth
# of 500 km for a range of temperatures. The velocity reduction
# $\Delta V / V = (V_{\text{el}} - V_{\text{anel}}) / V_{\text{el}}$
# quantifies the anelastic effect.

# +
test_depth = 500e3
test_temps = np.array([1500.0, 2000.0, 2500.0, 3000.0])
test_depths = np.full_like(test_temps, test_depth)

# Elastic velocities
vs_elastic = slb_pyrolite.temperature_to_vs(
    temperature=test_temps, depth=test_depths)

# Anelastic velocities for each model
anelastic_vs = {}
anelastic_vp = {}
velocity_reduction = {}

for model_key, corrected_model in all_models.items():
    vs_anel = corrected_model.temperature_to_vs(
        temperature=test_temps, depth=test_depths)
    vp_anel = corrected_model.temperature_to_vp(
        temperature=test_temps, depth=test_depths)
    anelastic_vs[model_key] = vs_anel
    anelastic_vp[model_key] = vp_anel
    reduction = (vs_elastic - vs_anel) / vs_elastic * 100
    velocity_reduction[model_key] = reduction

print(f"Elastic Vs at {test_depth / 1e3:.0f} km: {vs_elastic}")
for key in sorted(velocity_reduction):
    print(f"  {key}: reduction = {velocity_reduction[key]} %")
# -

# Visualisation
# -------------
#
# Figure 1 shows elastic vs anelastic Vs contour maps using the
# Cammarano Q3 model as a representative example. Figure 2 compares
# velocity reduction across all 8 models as a function of temperature.

# + tags=["active-ipynb"]
# %matplotlib inline
# -

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# # --- Figure 1: Elastic vs Anelastic Vs contours (Cammarano Q3) ---
# plt.close("all")
# fig, axes = plt.subplots(figsize=(10, 8), ncols=2)
# axes[0].set_position([0.08, 0.1, 0.35, 0.8])
# axes[1].set_position([0.44, 0.1, 0.35, 0.8])
#
# depths_x, temperatures_x = np.meshgrid(
#     slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(),
#     indexing="ij")
#
# cntr_lines = np.linspace(4000, 7000, 20)
# elastic_table = slb_pyrolite.compute_swave_speed()
# anelastic_table = cammarano_models["Cammarano_Q3"].compute_swave_speed()
#
# for idx, (table, title) in enumerate([
#         (elastic_table, "Elastic"),
#         (anelastic_table, "Anelastic (Cammarano Q3)")]):
#     axes[idx].contourf(
#         temperatures_x, depths_x, table.get_vals(), cntr_lines,
#         cmap=plt.colormaps["autumn"].resampled(20), extend="both")
#     axes[idx].invert_yaxis()
#     axes[idx].set_xlabel("Temperature [K]")
#     axes[idx].set_title(title)
#     axes[idx].grid(alpha=0.3)
#
# axes[0].set_ylabel("Depth [m]")
# axes[1].set_yticklabels([])
# plt.tight_layout()
# plt.show()
# -

# + tags=["active-ipynb"]
# # --- Figure 2: Velocity reduction for all models at 500 km ---
# plt.close("all")
# fig, ax = plt.subplots(figsize=(10, 6))
#
# dense_temps = np.linspace(1200, 3500, 200)
# dense_depths = np.full_like(dense_temps, test_depth)
# vs_el = slb_pyrolite.temperature_to_vs(
#     temperature=dense_temps, depth=dense_depths)
#
# for model_key in sorted(all_models):
#     vs_an = all_models[model_key].temperature_to_vs(
#         temperature=dense_temps, depth=dense_depths)
#     reduction = (vs_el - vs_an) / vs_el * 100
#     style = "-" if "Cammarano" in model_key else "--"
#     ax.plot(dense_temps, reduction, style, label=model_key, linewidth=1.5)
#
# ax.set_xlabel("Temperature [K]")
# ax.set_ylabel("Vs Reduction [%]")
# ax.set_title(f"Velocity Reduction at {test_depth/1e3:.0f} km Depth")
# ax.legend(ncol=2, fontsize=9)
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated how to:
# - Load an elastic thermodynamic model with `gdrift.ThermodynamicModel`
# - Create anelasticity models using `from_q_profile()` factory methods
#   for both Cammarano (Q1--Q6) and Goes (Q4, Q6) parameterizations
# - Apply anelastic corrections with `gdrift.apply_anelastic_correction()`
# - Compare velocity reductions across different Q-profiles
#
# The anelastic corrections are largest at high temperatures (near the
# solidus) and vary significantly between Q-profiles, reflecting
# uncertainties in mantle grain size and attenuation mechanisms.
