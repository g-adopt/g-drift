# Loading and Visualising Seismic Tomography Models
# ==================================================
#
# This example demonstrates how to load and visualise 3D seismic tomography
# models available in gdrift. Tomography models represent velocity perturbations
# (dVs, dVp) throughout the mantle, derived from seismic wave travel times.
#
# Background
# ----------
#
# Seismic tomography is a technique for imaging Earth's interior by analysing
# differences in seismic wave travel times. Faster velocities (positive dVs/Vs)
# typically indicate colder material, while slower velocities (negative dVs/Vs)
# suggest hotter regions. These velocity anomalies are key inputs for inferring
# mantle temperature structure in geodynamic studies.
#
# The gdrift package provides access to 25+ published tomography models,
# including widely-used models like S40RTS, GLAD-M25, and SEMUCB-WM1. Each
# model is stored as an HDF5 file with coordinates and velocity perturbation
# fields (dvs for shear waves, dvp for compressional waves).
#
# This example
# ------------
#
# We demonstrate how to:
# 1. List all available seismic tomography models
# 2. Identify which models are global and contain shear-wave velocity (Vs)
# 3. Query models at a specific depth (2700 km, near the core-mantle boundary)
# 4. Visualise all global Vs models on a Mollweide projection

import numpy as np
import gdrift

# +
# First, let's see all available seismic tomography models:

print("Available seismic tomography models:")
for model_name in gdrift.AVAILABLE_SEISMIC_MODELS:
    print(f"  - {model_name}")
# -

# Identifying Global Models with Shear-Wave Velocity
# --------------------------------------------------
#
# Not all tomography models cover the entire globe, and some only contain
# P-wave (compressional) velocities. We need to identify which models:
# 1. Have shear-wave velocity data (dvs field)
# 2. Cover the full globe (latitude from -90 to 90, longitude from -180 to 180)
#
# We do this by loading each model and checking its properties.


def get_global_vs_models():
    """
    Identify seismic tomography models that are global and contain Vs data.

    Returns a list of tuples: (model_name, model_object)
    """
    global_vs_models = []

    for model_name in gdrift.AVAILABLE_SEISMIC_MODELS:
        try:
            model = gdrift.SeismicModel(model_name)

            # Check if the model has dvs (shear-wave velocity perturbation)
            if not model.check_quantity("dvs"):
                print(f"  {model_name}: No dvs field, skipping")
                continue

            # Check if the model is global by examining coordinate extent
            coords = model.coordinates
            lat, lon, depth = gdrift.cartesian_to_geodetic(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )

            lat_range = (lat.min(), lat.max())
            lon_range = (lon.min(), lon.max())

            # Consider a model global if it spans most of lat/lon range
            is_global = (
                lat_range[0] < -80 and lat_range[1] > 80 and
                lon_range[0] < -170 and lon_range[1] > 170
            )

            if is_global:
                print(f"  {model_name}: Global Vs model "
                      f"(lat: {lat_range[0]:.1f} to {lat_range[1]:.1f}, "
                      f"lon: {lon_range[0]:.1f} to {lon_range[1]:.1f})")
                global_vs_models.append((model_name, model))
            else:
                print(f"  {model_name}: Regional model, skipping "
                      f"(lat: {lat_range[0]:.1f} to {lat_range[1]:.1f}, "
                      f"lon: {lon_range[0]:.1f} to {lon_range[1]:.1f})")

        except Exception as e:
            print(f"  {model_name}: Error loading - {e}")

    return global_vs_models


# +
print("\nScanning models for global coverage and Vs data:")
global_vs_models = get_global_vs_models()
print(f"\nFound {len(global_vs_models)} global Vs models")
# -

# Generating Query Points at a Fixed Depth
# ----------------------------------------
#
# We create a regular grid in latitude and longitude at 2700 km depth,
# which is close to the core-mantle boundary. This depth is of particular
# interest because it shows features like Large Low Shear Velocity Provinces
# (LLSVPs) beneath Africa and the Pacific.

# Define the grid parameters
depth = 2700e3  # 2700 km in meters
lat_resolution = 1  # 1 degree
lon_resolution = 1  # 1 degree

# Create the lat/lon grid
lats = np.arange(-90, 90 + lat_resolution, lat_resolution)
lons = np.arange(-180, 180 + lon_resolution, lon_resolution)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Convert to Cartesian coordinates for querying
depth_grid = np.full_like(lat_grid, depth)
query_coords = gdrift.geodetic_to_cartesian(lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel())

print(f"Query grid: {len(lats)} x {len(lons)} = {len(query_coords)} points at {depth/1e3:.0f} km depth")

# Querying All Global Vs Models
# -----------------------------
#
# Now we query each model at our grid points. The `at()` method returns
# interpolated values using the model's KD-tree with inverse distance weighting.

# +
model_data = {}
for model_name, model in global_vs_models:
    print(f"Querying {model_name}...")
    dvs = model.at("dvs", query_coords)
    dvs_grid = dvs.reshape(lat_grid.shape)
    model_data[model_name] = {
        "dvs": dvs_grid,
        "alpha": np.nanmax(np.abs(dvs_grid))  # max absolute value for colorbar scaling
    }
    print(f"  dVs range: {np.nanmin(dvs_grid):.2f}% to {np.nanmax(dvs_grid):.2f}%")
# -

# Visualisation
# -------------
#
# We plot all models using a Mollweide projection, which is well-suited for
# global data visualisation. Each subplot shows one model with its name and
# the maximum amplitude (alpha) annotated. All subplots share the same
# colormap but have individually scaled color ranges.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
#
# n_models = len(model_data)
# fig_height = 3 * n_models  # 3 inches per model
# fig, axes = plt.subplots(
#     n_models, 1,
#     figsize=(10, fig_height),
#     subplot_kw={"projection": "mollweide"}
# )
#
# # Handle single model case
# if n_models == 1:
#     axes = [axes]
#
# # Use a diverging colormap centred at zero
# cmap = plt.cm.RdBu
#
# for idx, (model_name, data) in enumerate(model_data.items()):
#     ax = axes[idx]
#     dvs = data["dvs"]
#     alpha = data["alpha"]
#
#     # Convert lon/lat to radians for Mollweide projection
#     lon_rad = np.radians(lon_grid)
#     lat_rad = np.radians(lat_grid)
#
#     # Create a symmetric normalisation around zero
#     norm = TwoSlopeNorm(vmin=-alpha, vcenter=0, vmax=alpha)
#
#     # Plot the data
#     im = ax.pcolormesh(lon_rad, lat_rad, dvs, cmap=cmap, norm=norm, shading="auto")
#
#     # Add gridlines
#     ax.grid(True, alpha=0.3)
#
#     # Title with model name and alpha value
#     ax.set_title(f"{model_name}  (±{alpha:.2f}%)", fontsize=12, fontweight="bold")
#
# # Add a shared colorbar
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(
#     plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
#     cax=cbar_ax,
#     orientation="vertical"
# )
# cbar.set_label(r"$\delta V_s / V_s$ (normalised to $\pm\alpha$)", fontsize=11)
# cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
# cbar.set_ticklabels([r"$-\alpha$", r"$-\alpha/2$", "0", r"$+\alpha/2$", r"$+\alpha$"])
#
# fig.suptitle(f"Global Seismic Tomography Models at {depth/1e3:.0f} km Depth", fontsize=14, y=0.98)
# plt.tight_layout(rect=[0, 0, 0.9, 0.96])
# plt.show()
# -

# Summary
# -------
#
# This example showed how to:
# - Access the list of available seismic tomography models via `gdrift.AVAILABLE_SEISMIC_MODELS`
# - Load models using `gdrift.SeismicModel(model_name)`
# - Check model properties (available fields, coordinate extent)
# - Query models at arbitrary points using `model.at("dvs", coordinates)`
# - Visualise global models on a Mollweide projection
#
# The velocity perturbations at 2700 km depth reveal the Large Low Shear
# Velocity Provinces (LLSVPs) beneath Africa and the Pacific, as well as
# faster regions associated with subducted slabs.
