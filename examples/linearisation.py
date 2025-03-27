import gdrift
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Tutorial: Linearising and Regularising Thermodynamic Tables
#
# This tutorial demonstrates how to regularise thermodynamic properties of
# Earth's mantle using `gdrift`, with a focus on linearising wave speeds
# (S-wave and P-wave) and density (ρ). The process involves building a
# regularised thermodynamic model, comparing original and regularised tables,
# and visualising the results at specific depths.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. Initial Setup
# Define the thermodynamic model for pyrolite composition.
# -----------------------------------------------------------------------------
slb_pyrolite = gdrift.ThermodynamicModel(
    "SLB_16", "pyrolite", temps=np.linspace(300, 4000), depths=np.linspace(0, 2890e3)
)

# -----------------------------------------------------------------------------
# 2. Building a Temperature Profile
# Create a `SplineProfile` to represent a depth-dependent temperature profile.
# This profile serves as a baseline for regularising the thermodynamic tables.
# -----------------------------------------------------------------------------
temperature_spline = gdrift.SplineProfile(
    depth=np.asarray([0., 500e3, 2700e3, 3000e3]),
    value=np.asarray([300, 1000, 3000, 4000])
)

# -----------------------------------------------------------------------------
# 3. Regularising the Thermodynamic Table
# Use the `regularise_thermodynamic_table` function to produce a corrected
# version of the model, ensuring the properties align with the temperature
# profile.
# -----------------------------------------------------------------------------
regular_slb_pyrolite = gdrift.regularise_thermodynamic_table(
    slb_pyrolite, temperature_spline, regular_range={"v_s": (-1.5, 0.0), "v_p": (-np.inf, 0.0), "rho": (-np.inf, 0.0)})

# -----------------------------------------------------------------------------
# 4. Extracting Data
# Extract the original and regularised tables for S-wave speed, P-wave speed,
# and density.
# -----------------------------------------------------------------------------
Vs_original = slb_pyrolite.compute_swave_speed().get_vals()
Vp_original = slb_pyrolite.compute_pwave_speed().get_vals()
rho_original = slb_pyrolite._tables["rho"].get_vals()

Vs_corrected = regular_slb_pyrolite.compute_swave_speed().get_vals()
Vp_corrected = regular_slb_pyrolite.compute_pwave_speed().get_vals()
rho_corrected = regular_slb_pyrolite._tables["rho"].get_vals()

# -----------------------------------------------------------------------------
# 5. Visualising the Results
# Visualise the S-wave speed, P-wave speed, and density for both the original
# and regularised models at specific depths. Results are presented in three
# columns, one for each property.
# -----------------------------------------------------------------------------
depths = np.asarray([410, 660, 1000, 2000]) * 1e3
indices = [abs(d - slb_pyrolite.get_depths()).argmin() for d in depths]

# Create figure with 3 columns for Vs, Vp, and rho
fig, axs = plt.subplots(len(indices), 3, figsize=(15, 10), constrained_layout=True)

# Labels for the columns
column_titles = [r"$v_s$", r"$v_p$", r"$\rho$"]

# Plotting data for each depth
for i, idx in enumerate(indices):
    depth = slb_pyrolite.get_depths()[idx]
    for j, (original, corrected, label) in enumerate(
        zip(
            [Vs_original, Vp_original, rho_original],
            [Vs_corrected, Vp_corrected, rho_corrected],
            column_titles,
        )
    ):
        axs[i, j].plot(slb_pyrolite.get_temperatures(), original[idx, :], color="blue", label="Original")
        axs[i, j].plot(slb_pyrolite.get_temperatures(), corrected[idx, :], color="red", label="Regularised")
        axs[i, j].axvline(
            x=temperature_spline.at_depth(depth), color="green", linestyle="--", label="Temperature Anchor"
        )
        axs[i, j].set_title(f"{label} at Depth: {depth / 1e3:.0f} km")
        axs[i, j].set_xlabel("Temperature (K)")
        axs[i, j].grid()
        if i == 0:
            axs[i, j].legend()

plt.show()
