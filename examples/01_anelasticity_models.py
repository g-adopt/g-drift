import matplotlib.pyplot as plt

"""
This script demonstrates how to apply anelastic correction to an existing thermodynamic table using the functionalities provided by the `gdrift` library. The anelastic model used in this example is based on the work of Cammarano et al., and the solidus curve is constructed by combining datasets from Andrault et al. (2011, EPSL) and Hirschmann (2000, G3).

Functions:
    build_solidus():
        Constructs the solidus curve for the mantle by combining the solidus curves from Andrault et al. (2011, EPSL) and Hirschmann (2000).

    build_anelasticity_model(solidus, q_profile: str = "Q1"):
        Constructs the anelasticity model based on the solidus curve and the parameters from Cammarano et al. The `q_profile` parameter allows selecting different sets of parameters.

    Regularising the table:
        Regularising the thermodynamic table involves adjusting the table to ensure that the seismic speeds are consistent at specific temperatures. This is done using a temperature profile representing the mantle average temperature.

    The `regularise_thermodynamic_table` function in `mineralogy.py`:
        This function adjusts the thermodynamic table to ensure consistency in seismic speeds at specific temperatures. It takes the original thermodynamic model, a temperature profile, and a range for regularisation as inputs and returns a regularised thermodynamic model.

Example Usage:
    The script demonstrates the following steps:
    1. Load the Preliminary Reference Earth Model (PREM).
    2. Create a thermodynamic model for pyrolite.
    3. Compute the elastic shear-wave and compressional-wave speeds.
    4. Build the solidus model.
    5. Construct the anelasticity model using the solidus model.
    6. Apply the anelastic correction to the thermodynamic model.
    7. Compute the anelastic shear-wave and compressional-wave speeds.
    8. Regularise the thermodynamic table.
    9. Apply the anelastic correction to the regularised thermodynamic table.
    10. Plot the results, including contour plots of shear-wave speeds and specific depth profiles for shear and compressional seismic speeds.
"""
import numpy as np
import gdrift

# In this tutorial we show how with the given functionalities
# we can apply anelastic correction to an existing thermodynamic table

# In this example, we use the anelastic model from Cammarano et al.
# The anelastic parameterization depends on the mantle's solidus curve.
# We combine two datasets to build this curve:
# - Andrault et al. (2011, EPSL)
# - Hirschmann (2000, G3)

# Load PREM
prem = gdrift.PreliminaryRefEarthModel()

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")
pyrolite_elastic_s_speed = slb_pyrolite.compute_swave_speed()
pyrolite_elastic_p_speed = slb_pyrolite.compute_pwave_speed()

# building solidus model
cammarano_q_model = "Q1"  # choose model from cammarano et al., 2003
anelasticity = gdrift.CammaranoAnelasticityModel.from_q_profile(cammarano_q_model)
anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(slb_pyrolite, anelasticity)

pyrolite_anelastic_s_speed = anelastic_slb_pyrolite.compute_swave_speed()
pyrolite_anelastic_p_speed = anelastic_slb_pyrolite.compute_pwave_speed()

# A temperautre profile representing the mantle average temperature
# This is used to anchor the regularised thermodynamic table (we make sure the seismic speeds are the same at those temperature for the regularised and unregularised table)
temperature_spline = gdrift.SplineProfile(
    depth=np.asarray([0.0, 500e3, 2700e3, 3000e3]), value=np.asarray([300, 1000, 3000, 4000])
)


# Regularising the table
# Regularisation works by saturating the minimum and maximum of variable gradients with respect to temperature.
# Default values are between -inf and 0.0; which essentialy prohibits phase jumps that would otherwise render
# v_s/v_p/rho versus temperature non-unique.
linear_slb_pyrolite = gdrift.mineralogy.regularise_thermodynamic_table(
    slb_pyrolite, temperature_spline, regular_range={"v_s": [-0.5, 0], "v_p": [-0.5, 0.0], "rho": [-0.5, 0.0]}
)

# Regularising the table
linear_anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(linear_slb_pyrolite, anelasticity)

# linearised seismic speeds
linear_pyrolite_anelastic_s_speed = linear_anelastic_slb_pyrolite.compute_swave_speed()
linear_pyrolite_anelastic_p_speed = linear_anelastic_slb_pyrolite.compute_pwave_speed()

# contour lines to plot
cntr_lines = np.linspace(4000, 7000, 20)

plt.close("all")
fig, axes = plt.subplots(figsize=(12, 8), ncols=3)
axes[0].set_position([0.05, 0.1, 0.25, 0.8])
axes[1].set_position([0.31, 0.1, 0.25, 0.8])
axes[2].set_position([0.57, 0.1, 0.25, 0.8])
# Getting the coordinates
depths_x, temperatures_x = np.meshgrid(slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")
img = []

for id, table in enumerate([pyrolite_elastic_s_speed, pyrolite_anelastic_s_speed, linear_pyrolite_anelastic_s_speed]):
    img.append(
        axes[id].contourf(
            temperatures_x,
            depths_x,
            table.get_vals(),
            cntr_lines,
            cmap=plt.colormaps["autumn"].resampled(20),
            extend="both",
        )
    )
    axes[id].invert_yaxis()
    axes[id].set_xlabel("Temperature [K]")
    axes[id].set_ylabel("Depth [m]")
    axes[id].grid()

for ax in axes[1:]:
    ax.set_ylabel("")
    ax.set_yticklabels("")

axes[0].text(
    0.5, 1.05, s="Elastic", transform=axes[0].transAxes, ha="center", va="center", bbox=dict(facecolor=(1.0, 1.0, 0.7))
)
axes[1].text(
    0.5,
    1.05,
    s="With Anelastic Correction",
    ha="center",
    va="center",
    transform=axes[1].transAxes,
    bbox=dict(facecolor=(1.0, 1.0, 0.7)),
)
axes[2].text(
    0.5,
    1.05,
    s="Linearised With Anelastic Correction",
    ha="center",
    va="center",
    transform=axes[2].transAxes,
    bbox=dict(facecolor=(1.0, 1.0, 0.7)),
)

fig.colorbar(
    img[-1],
    ax=axes[0],
    cax=fig.add_axes([0.84, 0.1, 0.02, 0.8]),
    orientation="vertical",
    label="Shear-Wave Speed [m/s]",
)


# Figure 2:
# Looking at a specific depth of shear seismic speed
plt.close(2)
fig_2 = plt.figure(num=2)
ax_2 = fig_2.add_subplot(111)
index = 150
ax_2.plot(
    pyrolite_anelastic_s_speed.get_y(),
    pyrolite_anelastic_s_speed.get_vals()[index, :],
    color="blue",
    label="With Anelastic Correction",
)
ax_2.plot(
    pyrolite_anelastic_s_speed.get_y(),
    linear_pyrolite_anelastic_s_speed.get_vals()[index, :],
    color="green",
    label="Linear Anelastic Model",
)
ax_2.plot(
    pyrolite_anelastic_s_speed.get_y(),
    pyrolite_elastic_s_speed.get_vals()[index, :],
    color="red",
    label="Elastic Model",
)
ax_2.vlines(
    [anelasticity.solidus.at_depth(pyrolite_anelastic_s_speed.get_x()[index])],
    ymin=pyrolite_anelastic_s_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_s_speed.get_vals()[index, :].max(),
    color="grey",
    label="Solidus",
    alpha=0.5,
)

ax_2.set_xlabel("Temperature[K]")
ax_2.set_ylabel("Shear Seismic-Wave Speed [m/s]")
ax_2.text(
    0.5,
    1.05,
    s=f"cammarano et al. {cammarano_q_model} at depth {pyrolite_anelastic_s_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center",
    va="center",
    transform=ax_2.transAxes,
    bbox=dict(facecolor=(1.0, 1.0, 0.7)),
)
ax_2.legend()
ax_2.grid()
plt.show()


# Figure 3:
# Looking at a specific depth of compressional seismic speed
plt.close(3)
fig_3 = plt.figure(num=3)
ax_3 = fig_3.add_subplot(111)
ax_3.plot(
    pyrolite_anelastic_p_speed.get_y(),
    pyrolite_anelastic_p_speed.get_vals()[index, :],
    color="blue",
    label="With Anelastic Correction",
)
ax_3.plot(
    pyrolite_anelastic_p_speed.get_y(),
    linear_pyrolite_anelastic_p_speed.get_vals()[index, :],
    color="green",
    label="Linear Anelastic Model",
)
ax_3.plot(
    pyrolite_anelastic_p_speed.get_y(),
    pyrolite_elastic_p_speed.get_vals()[index, :],
    color="red",
    label="Elastic Model",
)
ax_3.vlines(
    [anelasticity.solidus.at_depth(pyrolite_anelastic_p_speed.get_x()[index])],
    ymin=pyrolite_anelastic_p_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_p_speed.get_vals()[index, :].max(),
    color="grey",
    label="Solidus",
    alpha=0.5,
)

ax_3.set_xlabel("Temperature[K]")
ax_3.set_ylabel("Compressional Seismic-Wave Speed [m/s]")
ax_3.text(
    0.5,
    1.05,
    s=f"cammarano et al. {cammarano_q_model} at depth {pyrolite_anelastic_p_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center",
    va="center",
    transform=ax_3.transAxes,
    bbox=dict(facecolor=(1.0, 1.0, 0.7)),
)
ax_3.legend()
ax_3.grid()
plt.show()
