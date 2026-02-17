# Loading Solidus Temperature Profiles
# =====================================
#
# This example demonstrates how to load predefined solidus temperature
# profiles and create composite profiles spanning different depth ranges.
#
# Background
# ----------
#
# The solidus defines the temperature at which rock begins to melt at a
# given pressure (depth). Different experimental studies have constrained
# solidus temperatures for different depth ranges: Hirschmann (2000) for
# the upper mantle, Andrault et al. (2011) and Fiquet et al. (2010) for
# the deep mantle. These solidus profiles are essential inputs for
# anelastic velocity corrections in geodynamic modelling.
#
# This example
# ------------
#
# We demonstrate how to:
# 1. Load predefined solidus profiles from gdrift
# 2. Combine profiles to create a composite solidus spanning the full mantle
# 3. Visualise and compare different solidus models

import numpy as np
import gdrift
from gdrift.profile import SplineProfile

# Loading Predefined Solidus Models
# ----------------------------------
#
# gdrift provides several experimentally-derived solidus profiles as HDF5
# datasets. We load them using `RadialEarthModelFromFile` and access the
# solidus temperature profile via the `get_profile()` method.

# +
# Fiquet et al. (2010) solidus for the deep mantle
fiquet_solidus = gdrift.RadialEarthModelFromFile(
    model_name="1d_solidus_Fiquet_et_al_2010_SCIENCE",
    description="Fiquet et al 2010 Science")

# Andrault et al. (2011) solidus for the lower mantle
andrault_solidus = gdrift.RadialEarthModelFromFile(
    model_name="1d_solidus_Andrault_et_al_2011_EPSL",
    description="Andrault et al 2011 EPSL")

# Hirschmann (2000) solidus for the upper mantle (computed from pressure)
hirsch_solidus = gdrift.HirschmannSolidus()
# -

# Let's examine the depth ranges of each model:

# +
for name, model in [("Hirschmann", hirsch_solidus),
                    ("Andrault", andrault_solidus),
                    ("Fiquet", fiquet_solidus)]:
    profile = model.get_profile("solidus temperature")
    d_min, d_max = profile.min_max_depth()
    print(f"{name}: {d_min / 1e3:.0f} km to {d_max / 1e3:.0f} km")
# -

# Creating a Composite Solidus
# ----------------------------
#
# Different solidus models are valid over different depth ranges. A common
# approach is to combine the Hirschmann solidus (upper mantle) with a deep
# mantle solidus (Andrault or Fiquet) to create a continuous profile.
#
# Here we combine Hirschmann (upper mantle) with Andrault (lower mantle):

# +
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

# Create a new SplineProfile from the combined data
ghelichkhan_et_al = SplineProfile(
    depth=np.asarray(my_depths),
    value=np.asarray(my_solidus),
    name="solidus temperature (Ghelichkhan et al 2021)")

d_min, d_max = ghelichkhan_et_al.min_max_depth()
print(f"Composite solidus spans: {d_min / 1e3:.0f} km to {d_max / 1e3:.0f} km")
# -

# Visualisation
# -------------
#
# We plot all four solidus profiles to compare them. The composite profile
# (Ghelichkhan et al.) shows the smooth transition between upper and lower
# mantle solidus temperatures.

# + tags=["active-ipynb"]
# %matplotlib inline
# -

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# plt.close(1)
# fig = plt.figure(num=1, figsize=(8, 6))
# ax = fig.add_subplot(111)
#
# for solidus_model, marker in zip(
#         [andrault_solidus.get_profile("solidus temperature"),
#          hirsch_solidus.get_profile("solidus temperature"),
#          fiquet_solidus.get_profile("solidus temperature"),
#          ghelichkhan_et_al],
#         ["-", "-.", "--", ":"]):
#     d_min, d_max = solidus_model.min_max_depth()
#     dpths = np.arange(d_min, d_max, 10e3)
#     # Use display_name for richer legend labels
#     ax.plot(solidus_model.at_depth(dpths), dpths / 1e3,
#             linestyle=marker, linewidth=2, label=solidus_model.display_name)
#
# ax.grid(alpha=0.3)
# ax.invert_yaxis()
# ax.set_xlabel("Solidus Temperature [K]", fontsize=12)
# ax.set_ylabel("Depth [km]", fontsize=12)
# ax.legend(loc="lower left")
# ax.set_title("Mantle Solidus Temperature Profiles")
# plt.tight_layout()
# plt.show()
# -

# Summary
# -------
#
# This example demonstrated how to:
# - Load predefined solidus profiles using `gdrift.RadialEarthModelFromFile`
# - Use the `HirschmannSolidus` model for upper mantle solidus temperatures
# - Combine multiple profiles into a composite solidus using `SplineProfile`
# - Query solidus temperature at arbitrary depths via `profile.at_depth()`
#
# The solidus profiles loaded here can be used as inputs for anelastic
# velocity corrections (see the anelasticity example).
