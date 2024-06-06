import matplotlib.pyplot as plt
import numpy as np
import gdrift
from gdrift.profile import RadialProfileSpline


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
prem = gdrift.PreRefEarthModel()

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
# axes[0]
depths_x, temperatures_x = np.meshgrid(
    slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")

for id, table in enumerate([pyrolite_anelastic_speed, pyrolite_elastic_speed]):
    axes[id].contourf(temperatures_x, depths_x,
                      table.get_vals(), cntr_lines, cmap=cm.get_cmap(""))

fig.show()

plt.close(2)
fig_2 = plt.figure(num=2)
ax_2 = fig_2.add_subplot(111)
ax_2.plot(pyrolite_anelastic_speed.get_y(), pyrolite_anelastic_speed.get_vals()[100, :])
ax_2.plot(pyrolite_anelastic_speed.get_y(), pyrolite_elastic_speed.get_vals()[100, :])
fig_2.show()
