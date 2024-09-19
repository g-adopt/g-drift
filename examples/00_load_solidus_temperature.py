"""This example shows the way to load predefined solidus
   temperatures, or definiing your own.
"""
import gdrift
from gdrift.profile import RadialProfileSpline
import numpy as np
import matplotlib.pyplot as plt

# Defining the solidus curve for manlte
fiquet_solidus = gdrift.SolidusProfileFromFile(
    model_name="1d_solidus_Fiquet_et_al_2010_SCIENCE",
    description="Fiquet et al 2010 Science")

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

ghelichkhan_et_al = RadialProfileSpline(
    depth=np.asarray(my_depths),
    value=np.asarray(my_solidus),
    name="Ghelichkhan et al 2021")

plt.close(1)
fig = plt.figure(num=1)
ax = fig.add_subplot(111)
for solidus_model, marker in zip(
        [andrault_solidus, hirsch_solidus, fiquet_solidus, ghelichkhan_et_al],
        ["-", "-.", "--", ":"]):
    d_min, d_max = solidus_model.min_max_depth()
    dpths = np.arange(d_min, d_max, 10e3)
    ax.plot(solidus_model.at_depth(dpths), dpths/1e3,
            linestyle=marker, label=solidus_model.name)

ax.grid()
ax.invert_yaxis()
ax.set_xlabel("Solidus Temperature [K]")
ax.set_ylabel("Depth [km]")
ax.legend()
fig.show()
