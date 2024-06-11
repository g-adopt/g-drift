import matplotlib.pyplot as plt
import numpy as np
import gdrift
from gdrift.profile import RadialProfileSpline

# In this tutorial we show how with the given functionalities
# we can apply anelastic correction to an existing thermodynamic table


# For our anelastic model in this example we use the study by Cammerano et al.


def build_solidus():
    # Defining the solidus curve for manlte
    andrault_solidus = gdrift.SolidusProfileFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Defining parameters for Cammarano style anelasticity model
    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []
    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    my_depths.extend([3000e3])
    my_solidus.extend([solidus_model.at_depth(dpths[-1])])

    ghelichkhan_et_al = RadialProfileSpline(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021")

    return ghelichkhan_et_al


def build_anelasticity_model(solidus):
    def B(x): return np.where(x < 660e3, 1.1, 20)
    def g(x): return np.where(x < 660e3, 20, 10)
    def a(x): return 0.2
    def omega(x): return 1.0
    return gdrift.CammaranoAnelasticityModel(B, g, a, solidus, omega)


# Load PREM
prem = gdrift.PreliminaryRefEarthModel()

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")
pyrolite_elastic_speed = slb_pyrolite.compute_swave_speed()

# building solidus model
solidus_ghelichkhan = build_solidus()
anelasticity = build_anelasticity_model(solidus_ghelichkhan)
anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
    slb_pyrolite, anelasticity)
pyrolite_anelastic_speed = anelastic_slb_pyrolite.compute_swave_speed()

# contour lines to plot
cntr_lines = np.linspace(4000, 7000, 20)

plt.close("all")
fig, axes = plt.subplots(ncols=2)
axes[0].set_position([0.1, 0.1, 0.35, 0.8])
axes[1].set_position([0.5, 0.1, 0.35, 0.8])
# Getting the coordinates
depths_x, temperatures_x = np.meshgrid(
    slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")
img = []

for id, table in enumerate([pyrolite_elastic_speed, pyrolite_anelastic_speed]):
    img.append(axes[id].contourf(
        temperatures_x,
        depths_x,
        table.get_vals(),
        cntr_lines,
        cmap=plt.colormaps["autumn"].resampled(20),
        extend="both"))
    axes[id].invert_yaxis()
    axes[id].set_xlabel("Temperature [K]")
    axes[id].set_ylabel("Depth [m]")
    axes[id].grid()

axes[1].set_ylabel("")
axes[1].set_yticklabels("")

axes[0].text(0.5, 1.05, s="Elastic", transform=axes[0].transAxes,
             ha="center", va="center",
             bbox=dict(facecolor=(1.0, 1.0, 0.7)))
axes[1].text(0.5, 1.05, s="With Anelastic Correction",
             ha="center", va="center",
             transform=axes[1].transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
fig.colorbar(img[-1], ax=axes[0], cax=fig.add_axes([0.88,
             0.1, 0.02, 0.8]), orientation="vertical", label="Shear-Wave Speed [m/s]")

plt.close(2)
fig_2 = plt.figure(num=2)
ax_2 = fig_2.add_subplot(111)
index = 100
ax_2.plot(pyrolite_anelastic_speed.get_y(),
          pyrolite_anelastic_speed.get_vals()[index, :], color="blue", label="With Anelastic Correction")
ax_2.plot(pyrolite_anelastic_speed.get_y(),
          pyrolite_elastic_speed.get_vals()[index, :], color="red", label="Elastic Model")
ax_2.vlines(
    [solidus_ghelichkhan.at_depth(pyrolite_anelastic_speed.get_x()[index])],
    ymin=pyrolite_anelastic_speed.get_vals()[index, :].min(),
    ymax=pyrolite_anelastic_speed.get_vals()[index, :].max(),
    color="grey", label="Solidus", alpha=0.5)

ax_2.set_xlabel("Temperature[K]")
ax_2.set_ylabel("Seismic-Wave Speed [m/s]")
ax_2.text(
    0.5, 1.05, s=f"At depth {pyrolite_anelastic_speed.get_x()[index]/1e3:.1f} [m]",
    ha="center", va="center",
    transform=ax_2.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))
ax_2.legend()
ax_2.grid()
plt.show()