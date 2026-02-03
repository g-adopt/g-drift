"""
This script demonstrates how to apply anelastic correction to an existing
thermodynamic table using the gdrift library.

The anelastic model is based on the work of Cammarano et al., and the solidus
curve is constructed by combining datasets from Andrault et al. (2011, EPSL)
and Hirschmann (2000, G3).

Steps:
    1. Load the Preliminary Reference Earth Model (PREM).
    2. Create a thermodynamic model for pyrolite.
    3. Compute elastic shear-wave and compressional-wave speeds.
    4. Build the solidus model and construct the anelasticity model.
    5. Apply the anelastic correction.
    6. Plot elastic vs. anelastic shear-wave speed contours.
"""
import matplotlib.pyplot as plt
import numpy as np
import gdrift
from gdrift.profile import SplineProfile
from gdrift.anelasticity import BaseAnelasticityModel


def build_solidus():
    """Construct the composite solidus for the mantle by combining
    Hirschmann (shallow) and Andrault (deep) profiles."""
    andrault_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al. 2011, EPSL")

    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []
    for solidus_model in [
        hirsch_solidus.get_profile("solidus temperature"),
        andrault_solidus.get_profile("solidus temperature"),
    ]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    # Extend to 3000 km to avoid extrapolation
    my_depths.extend([3000e3])
    my_solidus.extend([solidus_model.at_depth(dpths[-1])])

    return SplineProfile(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        extrapolate=True,
        name="Ghelichkhan et al 2021")


def build_anelasticity_model(solidus, q_profile: str = "Q1"):
    """Construct a Cammarano anelasticity model from a solidus and Q-profile."""
    cammarano_parameters = {
        "Q1": {"B": [0.5, 10], "g": [20, 10]},
        "Q2": {"B": [0.8, 15], "g": [20, 10]},
        "Q3": {"B": [1.1, 20], "g": [20, 10]},
        "Q4": {"B": [0.035, 2.25], "g": [30, 15]},
        "Q5": {"B": [0.056, 3.6], "g": [30, 15]},
        "Q6": {"B": [0.077, 4.95], "g": [30, 15]},
    }

    def B(x):
        return np.where(x < 660e3, cammarano_parameters[q_profile]["B"][0],
                        cammarano_parameters[q_profile]["B"][1])

    def g(x):
        return np.where(x < 660e3, cammarano_parameters[q_profile]["g"][0],
                        cammarano_parameters[q_profile]["g"][1])

    def a(x):
        return 0.2

    def omega(x):
        return 1.

    def Q_kappa(x):
        return np.where(x < 660e3, 1e3, 1e4)

    return gdrift.CammaranoAnelasticityModel(
        B=B, g=g, a=a, solidus=solidus, Q_bulk=Q_kappa, omega=omega)


# ------------------------------------------------------------------
# Load PREM
# ------------------------------------------------------------------
prem = gdrift.PreliminaryRefEarthModel()

# ------------------------------------------------------------------
# Thermodynamic model
# ------------------------------------------------------------------
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")
pyrolite_elastic_s_speed = slb_pyrolite.compute_swave_speed()
pyrolite_elastic_p_speed = slb_pyrolite.compute_pwave_speed()

# ------------------------------------------------------------------
# Build solidus + anelasticity and apply correction
# ------------------------------------------------------------------
solidus_ghelichkhan = build_solidus()
cammarano_q_model = "Q1"
anelasticity = build_anelasticity_model(solidus_ghelichkhan, q_profile=cammarano_q_model)
anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(slb_pyrolite, anelasticity)

pyrolite_anelastic_s_speed = anelastic_slb_pyrolite.compute_swave_speed()
pyrolite_anelastic_p_speed = anelastic_slb_pyrolite.compute_pwave_speed()

# ------------------------------------------------------------------
# Figure 1: contour plots — elastic vs anelastic Vs
# ------------------------------------------------------------------
cntr_lines = np.linspace(4000, 7000, 20)

plt.close("all")
fig, axes = plt.subplots(figsize=(10, 8), ncols=2)
axes[0].set_position([0.08, 0.1, 0.35, 0.8])
axes[1].set_position([0.44, 0.1, 0.35, 0.8])

depths_x, temperatures_x = np.meshgrid(
    slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")

img = []
for idx, table in enumerate([pyrolite_elastic_s_speed, pyrolite_anelastic_s_speed]):
    img.append(axes[idx].contourf(
        temperatures_x, depths_x, table.get_vals(),
        cntr_lines,
        cmap=plt.colormaps["autumn"].resampled(20),
        extend="both"))
    axes[idx].invert_yaxis()
    axes[idx].set_xlabel("Temperature [K]")
    axes[idx].set_ylabel("Depth [m]")
    axes[idx].grid()

axes[1].set_ylabel("")
axes[1].set_yticklabels("")

axes[0].text(0.5, 1.05, s="Elastic", transform=axes[0].transAxes,
             ha="center", va="center",
             bbox=dict(facecolor=(1.0, 1.0, 0.7)))
axes[1].text(0.5, 1.05, s="With Anelastic Correction",
             ha="center", va="center",
             transform=axes[1].transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))

fig.colorbar(img[-1], ax=axes[0], cax=fig.add_axes([0.82, 0.1, 0.02, 0.8]),
             orientation="vertical", label="Shear-Wave Speed [m/s]")

# ------------------------------------------------------------------
# Figure 2: depth slice — shear seismic speed
# ------------------------------------------------------------------
plt.close(2)
fig_2 = plt.figure(num=2)
ax_2 = fig_2.add_subplot(111)
index = 150
ax_2.plot(pyrolite_anelastic_s_speed.get_y(),
          pyrolite_anelastic_s_speed.get_vals()[index, :],
          color="blue", label="With Anelastic Correction")
ax_2.plot(pyrolite_anelastic_s_speed.get_y(),
          pyrolite_elastic_s_speed.get_vals()[index, :],
          color="red", label="Elastic Model")
ax_2.vlines(
    [solidus_ghelichkhan.at_depth(pyrolite_anelastic_s_speed.get_x()[index])],
    ymin=pyrolite_anelastic_s_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_s_speed.get_vals()[index, :].max(),
    color="grey", label="Solidus", alpha=0.5)

ax_2.set_xlabel("Temperature [K]")
ax_2.set_ylabel("Shear Seismic-Wave Speed [m/s]")
ax_2.text(
    0.5, 1.05,
    s=f"Cammarano et al. {cammarano_q_model} at depth "
      f"{pyrolite_anelastic_s_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center", va="center",
    transform=ax_2.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
ax_2.legend()
ax_2.grid()

# ------------------------------------------------------------------
# Figure 3: depth slice — compressional seismic speed
# ------------------------------------------------------------------
plt.close(3)
fig_3 = plt.figure(num=3)
ax_3 = fig_3.add_subplot(111)
ax_3.plot(pyrolite_anelastic_p_speed.get_y(),
          pyrolite_anelastic_p_speed.get_vals()[index, :],
          color="blue", label="With Anelastic Correction")
ax_3.plot(pyrolite_anelastic_p_speed.get_y(),
          pyrolite_elastic_p_speed.get_vals()[index, :],
          color="red", label="Elastic Model")
ax_3.vlines(
    [solidus_ghelichkhan.at_depth(pyrolite_anelastic_p_speed.get_x()[index])],
    ymin=pyrolite_anelastic_p_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_p_speed.get_vals()[index, :].max(),
    color="grey", label="Solidus", alpha=0.5)

ax_3.set_xlabel("Temperature [K]")
ax_3.set_ylabel("Compressional Seismic-Wave Speed [m/s]")
ax_3.text(
    0.5, 1.05,
    s=f"Cammarano et al. {cammarano_q_model} at depth "
      f"{pyrolite_anelastic_p_speed.get_x()[index] / 1e3:.1f} [km]",
    ha="center", va="center",
    transform=ax_3.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
ax_3.legend()
ax_3.grid()
plt.show()
