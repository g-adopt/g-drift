"""
Compute 1D geodynamic adiabatic profiles using the SLB 2021 pyrolite (CFMAS)
thermodynamic model.

This script:
    1. Loads PREM and computes gravity from density.
    2. Loads the SLB_21 pyroliteCFMAS ThermodynamicModel.
    3. Defines dT/dP(T, depth) using temperature_to_property for alpha, rho, V, Cp.
    4. Integrates the ODE with scipy.integrate.odeint from T0=1600 K at the surface to the CMB.
    5. Computes profiles (rho, alpha, Cp, V, Cv, beta, gamma) along the adiabat.
    6. Saves the result to HDF5 using create_dataset_file.
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

import gdrift
from gdrift.profile import SplineProfile

# ------------------------------------------------------------------
# 1. Load PREM and build a gravity-vs-depth spline
# ------------------------------------------------------------------
prem = gdrift.PreliminaryRefEarthModel()
prem_radius = np.linspace(0, gdrift.R_earth, 1000)
prem_depths = gdrift.R_earth - prem_radius
prem_mass = gdrift.compute_mass(
    radius=prem_radius,
    density=prem.at_depth("density", prem_depths),
)
prem_g = gdrift.compute_gravity(prem_radius, prem_mass)

depth2grav = SplineProfile(
    depth=prem_depths,
    value=prem_g,
    name="gravity",
    extrapolate=True,
)

# ------------------------------------------------------------------
# 2. Load thermodynamic model
# ------------------------------------------------------------------
slb_pyrolite = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")


# ------------------------------------------------------------------
# 3. Define the ODE: dT/dz along the adiabat
# ------------------------------------------------------------------
def dT_dP(T, depth):
    alpha = slb_pyrolite.temperature_to_property("alpha", T, depth)
    rho = slb_pyrolite.temperature_to_property("rho", T, depth)
    V = slb_pyrolite.temperature_to_property("V", T, depth)
    c_p = slb_pyrolite.temperature_to_property("Cp", T, depth)

    M = rho * V  # molar mass per unit volume -> Cp/M gives SI Cp
    print(f"At around {depth / 1e3:.1f} km.")
    return alpha * T * depth2grav.at_depth(depth) / (c_p / M)


# ------------------------------------------------------------------
# 4. Integrate from surface to CMB
# ------------------------------------------------------------------
geodynamic_profile_names = [
    "rho", "alpha", "Cp", "V", "Cv", "beta", "gamma",
]

T0 = 1600  # surface potential temperature [K]
depths = np.linspace(0, gdrift.R_earth - gdrift.R_cmb, 257)

profiles = {}
profiles["Depths"] = depths
profiles["Tbar"] = scipy.integrate.odeint(
    func=dT_dP, y0=T0, t=depths, rtol=1e-3,
).squeeze()

# ------------------------------------------------------------------
# 5. Compute material properties along the adiabat
# ------------------------------------------------------------------
for profile_name in geodynamic_profile_names:
    profiles[f"{profile_name}bar"] = slb_pyrolite.temperature_to_property(
        profile_name, profiles["Tbar"], depths,
    )

# Derived SI heat capacities and gravity
profiles["CvSIbar"] = profiles["Cvbar"] / (profiles["rhobar"] * profiles["Vbar"])
profiles["CpSIbar"] = profiles["Cpbar"] / (profiles["rhobar"] * profiles["Vbar"])
profiles["gbar"] = depth2grav.at_depth(depths)

# ------------------------------------------------------------------
# 6. Plot
# ------------------------------------------------------------------
x0 = 0.05
y0 = 0.05
del_x = 0.12
del_y = 0.42
mar_x = 0.005
mar_y = 0.07

plt.close(1)
fig = plt.figure(1, figsize=(16, 10))

for i, profile_name in enumerate(profiles.keys()):
    ax = fig.add_subplot(111)
    ax.set_position([
        x0 + (i % 6) * (del_x + mar_x),
        y0 + int(i / 6) * (del_y + mar_y),
        del_x, del_y,
    ])
    ax.plot(profiles[profile_name], depths)
    ax.grid()
    ax.set_xlabel(f"{profile_name}")
    if i % 6 != 0:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.invert_yaxis()

Di = (profiles["alphabar"][0] * profiles["gbar"][0]
      * (gdrift.R_earth - gdrift.R_cmb) / profiles["CpSIbar"][0])
title_txt = f"Di: {Di:.2f}\n"
title_txt += "\n".join(
    f"{k}= {profiles[k][0]:.2e}"
    for k in ["Tbar", "rhobar", "alphabar", "CpSIbar", "betabar", "gbar"]
)
ax.text(
    0.8, 0.8, s=title_txt, transform=fig.transFigure,
    ha="center", va="center",
    bbox=dict(facecolor=(1.0, 1.0, 0.7)),
)
plt.show()

# ------------------------------------------------------------------
# 7. Save to HDF5
# ------------------------------------------------------------------
meta_data = {
    "Author": "Sia Ghelichkhan (siavash.ghelichkhan@anu.edu.au)",
    "Description": "Profiles for adiabatic temperature at 1600 [K], using SLB21_pyroliteCFMAS",
    "Software": "using EoS_MMO software authored by Chust et al 2017",
}

gdrift.create_dataset_file(
    "1d_geodynamic_SLB21_pyroliteCFMAS.h5",
    profiles,
    metadata=meta_data,
)
