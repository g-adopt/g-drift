from gdrift import *
import numpy as np
import matplotlib.pyplot as plt

# Defining the solidus curve for manlte
figuet_solidus = SolidusProfileFromFile(
    model_name="1d_solidus_Fiquet_et_al_2010_SCIENCE")

# Defining the solidus curve for manlte
andrault_solidus = SolidusProfileFromFile(
    model_name="1d_solidus_Andrault_et_al_2011_EPSL")

# Defining parameters for Cammarano style anelasticity model
hirsch_solidus = HirschmannSolidus()

dpths = np.linspace(0, 2890e3, 100)

plt.close(1)
fig = plt.figure(num=1)
ax = fig.add_subplot(111)
hirsch_solidus.at_depth(dpths)
ax.plot(hirsch_solidus._depth_to_pressure(dpths), dpths)
# # for model in [figuet_solidus, andrault_solidus, hirsch_solidus]:
# for model in [ hirsch_solidus]:
#     (min_model, max_model) = model.min_max_depth()
#     valid_mask = (dpths >= min_model) & (dpths <= max_model)
#     valid_depths = dpths[valid_mask]
#     ax.plot(valid_depths, model.at_depth(valid_depths) )
ax.set_xscale("log")
fig.show()