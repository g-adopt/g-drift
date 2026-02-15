# Loading gdrift Fields onto a gadopt Mesh
# =========================================
#
# This example demonstrates how to integrate `gdrift` with `gadopt`
# (the G-ADOPT geodynamic modelling platform built on Firedrake) to
# load seismic tomography data onto a spherical finite-element mesh
# and convert it to a temperature field suitable for mantle convection
# simulations.
#
# Background
# ----------
#
# Large-scale mantle convection simulations often require initial
# temperature fields derived from seismic observations. The workflow
# involves:
#
# 1. Creating a spherical mesh discretising the mantle volume.
# 2. Loading a 3D seismic tomography model.
# 3. Interpolating seismic velocities onto the mesh nodes.
# 4. Converting velocities to temperature via a thermodynamic lookup
#    table with anelastic corrections.
#
# The `gdrift` library handles seismic model loading, thermodynamic
# tables, solidus profiles, and anelastic corrections. The `gadopt`
# library (which re-exports Firedrake via ``from gadopt import *``)
# provides the mesh infrastructure and finite-element function spaces.
#
# This example
# ------------
#
# We construct a cubed-sphere mesh, load the REVEAL seismic model,
# interpolate its anisotropic shear-wave velocities ($V_{SH}$ and
# $V_{SV}$) onto the mesh, compute the isotropic velocity
# $V_S = \sqrt{(2 V_{SH}^2 + V_{SV}^2) / 3}$, and convert $V_S$
# to temperature using a regularised SLB_21 thermodynamic model with
# Cammarano-style anelastic corrections.

from pathlib import Path

import numpy as np
from gadopt import *
import gdrift

_demo_dir = Path(__file__).parent

# Helper functions
# ----------------
#
# We define three helper functions that build the components needed
# for velocity-to-temperature conversion: a composite solidus profile,
# a Cammarano-style anelasticity model, and the full regularised +
# anelastically-corrected thermodynamic model.


# +
def build_solidus():
    """Build a composite solidus from Hirschmann and Andrault models."""
    andrault = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")
    hirschmann = gdrift.HirschmannSolidus()

    depths = []
    solidus_temps = []
    for model in [hirschmann, andrault]:
        d_min, d_max = model.min_max_depth("solidus temperature")
        d = np.arange(d_min, d_max, 10e3)
        depths.extend(d)
        solidus_temps.extend(model.at_depth("solidus temperature", d))

    return gdrift.SplineProfile(
        depth=np.asarray(depths),
        value=np.asarray(solidus_temps),
        name="Composite solidus",
        extrapolate=True)


def build_anelasticity_model(solidus):
    """Build a Cammarano-style anelasticity model with depth-dependent parameters."""
    def B(x): return np.where(x < 660e3, 1.1, 20)
    def g(x): return np.where(x < 660e3, 20, 10)
    def a(x): return 0.2
    def omega(x): return 1.0
    return gdrift.CammaranoAnelasticityModel(B, g, a, solidus, omega)


def build_thermodynamic_model():
    """Build a regularised, anelastically-corrected thermodynamic model.

    The pipeline:
    1. Load SLB_21 pyrolite (CFMAS) thermodynamic tables.
    2. Load a reference temperature profile from a Terra simulation.
    3. Regularise the tables along the reference profile to smooth
       phase-transition discontinuities.
    4. Apply Cammarano-style anelastic corrections.
    """
    slb = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")

    terra_data = np.loadtxt(
        _demo_dir.parent / "TerraMT512vs.dat", unpack=False, usecols=(0, 1))
    terra_profile = gdrift.SplineProfile(
        depth=terra_data[:, 0] * 1e3,
        value=terra_data[:, 1],
        name="TerraMT512vs",
        extrapolate=True)

    regular_slb = gdrift.regularise_thermodynamic_table(
        slb, terra_profile,
        regular_range={
            "v_s": (-1.5, 0.0),
            "v_p": (-np.inf, 0.0),
            "rho": (-np.inf, 0.0),
        })

    solidus = build_solidus()
    anelasticity = build_anelasticity_model(solidus)
    return gdrift.apply_anelastic_correction(regular_slb, anelasticity)
# -


# Creating the spherical mesh
# ----------------------------
#
# We use gadopt's `CubedSphereMesh` to create a 2D cubed-sphere surface
# at the CMB radius, then extrude it radially to create a 3D spherical
# shell mesh. The non-dimensional radii `rmin` and `rmax` are scaled
# later to physical Earth coordinates.

# +
rmin, rmax = 1.208, 2.208
ref_level, nlayers = 3, 4

mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial")
mesh.cartesian = False

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
print(f"Mesh created: {mesh.num_cells()} cells")
# -

# Mapping to physical coordinates
# ---------------------------------
#
# The mesh coordinates are non-dimensional. We scale them to physical
# Earth coordinates (metres) by multiplying by $R_{\oplus} / r_{\max}$.

# +
X = SpatialCoordinate(mesh)
r = Function(V, name="coordinates").interpolate(X / rmax * gdrift.R_earth)

depth = Function(Q, name="depth").interpolate(
    Constant(gdrift.R_earth) - sqrt(r[0]**2 + r[1]**2 + r[2]**2)
)
print(f"Depth range: {depth.dat.data_with_halos.min()/1e3:.0f} "
      f"- {depth.dat.data_with_halos.max()/1e3:.0f} km")
# -

# Loading the REVEAL seismic model
# ----------------------------------
#
# We load the REVEAL seismic tomography model and query it for the
# horizontally- and vertically-polarised shear-wave velocities
# ($V_{SH}$ and $V_{SV}$) at each mesh node.

# +
seismic_model = gdrift.SeismicModel("REVEAL")

vsh = Function(Q, name="vsh")
vsv = Function(Q, name="vsv")
vs = Function(Q, name="vs")

reveal_data = seismic_model.at(
    label=["vsh", "vsv"], coordinates=r.dat.data_with_halos)
vsh.dat.data_with_halos[:] = reveal_data[:, 0]
vsv.dat.data_with_halos[:] = reveal_data[:, 1]
# -

# Computing isotropic velocity
# ------------------------------
#
# The isotropic shear-wave speed is the Voigt average of the anisotropic
# components:
#
# $$V_S = \sqrt{\frac{2 V_{SH}^2 + V_{SV}^2}{3}}$$

# +
vs.interpolate(sqrt((2 * vsh**2 + vsv**2) / 3))

v_ave = Function(Q, name="v_ave")
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(v_ave, averager.get_layer_average(vs))
print(f"Vs range: {vs.dat.data_with_halos.min():.1f} "
      f"- {vs.dat.data_with_halos.max():.1f} m/s")
# -

# Converting velocity to temperature
# -------------------------------------
#
# Using the regularised and anelastically-corrected thermodynamic model,
# we convert the isotropic shear-wave speed to temperature at each mesh
# node. The conversion uses inverse interpolation of the $V_s(T, z)$
# lookup table.

# +
anelastic_model = build_thermodynamic_model()

temperature = Function(Q, name="temperature")
temperature.dat.data_with_halos[:] = anelastic_model.vs_to_temperature(
    vs.dat.data_with_halos, depth.dat.data_with_halos)

t_ave = Function(Q, name="average_temperature")
averager.extrapolate_layer_average(t_ave, averager.get_layer_average(temperature))
print(f"Temperature range: {temperature.dat.data_with_halos.min():.0f} "
      f"- {temperature.dat.data_with_halos.max():.0f} K")
# -

# Writing output files
# ----------------------
#
# We write the fields to a VTK file for visualisation in ParaView and
# store the temperature in an HDF5 checkpoint that can be loaded as an
# initial condition for adjoint inversions.

# + tags=["active-ipynb"]
# vtk_file = VTKFile("REVEAL.pvd")
# vtk_file.write(vs, vsh, vsv, v_ave, temperature, t_ave, depth)
#
# with CheckpointFile("REVEAL_temperature.h5", "w") as checkpoint:
#     checkpoint.save_mesh(mesh)
#     checkpoint.save_function(temperature, name="Temperature")
# -

# Summary
# -------
#
# This example demonstrated the full workflow for loading seismic
# tomography data onto a gadopt finite-element mesh:
#
# - Created a cubed-sphere mesh using gadopt/Firedrake
# - Loaded the REVEAL seismic model with `gdrift.SeismicModel`
# - Interpolated anisotropic velocities and computed isotropic $V_S$
# - Built a regularised thermodynamic model with anelastic corrections
# - Converted $V_S$ to temperature for use in convection simulations
