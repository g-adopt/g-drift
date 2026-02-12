# Loading a tomography model and plotting it
# ==========================================
#
# In this tutorial we show how a tomography model that is available in the `gdrift` library can be loaded and plotted.
#
# !!! Note
# Notice that the first time any data set is loaded in g-drift, a file is downloaded from the internet.
# Since g-drift is parallel agnostic, it is important that you are not running any of this in paralell.
#
# Let's begin by importing the necessary library, which is gdrift and matplotlib

# +
import gdrift
import numpy as np  # for generating coordinates
import matplotlib.pyplot as plt  # for plotting
import cartopy.crs as ccrs
# -

# We will be using the REVEAL model, which is a global full-waveform inversion model.
# Let's load the model

llnl = gdrift.SeismicModel("LLNL-G3Dv3")

# lons = np.linspace(-180, 180, 721)
# lats = np.linspace(-90, 90, 361)
# depths = np.linspace(0, gdrift.R_earth - gdrift.R_cmb, 65)
# lons_x, lats_x, depths_x = np.meshgrid(lons, lats, depths, indexing="ij")
#
# coordinates = gdrift.geodetic_to_cartesian(
#     lats_x.flatten(), lons_x.flatten(), depths_x.flatten()).T
#
# dVs = s40rts.at(label="dvs", coordinates=np.column_stack(coordinates)).reshape(depths_x.shape)
#
#
# with open("depth_layers.dat", "w") as f:
#     f.write("\n".join(f"{depth:4.0f}" for depth in depths))
#
# for ir, depth in enumerate(depths):
#     output_str = "\n".join(
#         f"{lon:+5.2f}, {lat:+5.2f}, {value:+5.2f}"
#         for lon, lat, value in zip(lons_x[..., ir].flatten(), lats_x[..., ir].flatten(), dVs[..., ir].flatten())
#     )
#     with open(f"s40rts.layer.{ir:03d}.dat", "w") as f:
#         f.write(output_str)
#
#
#
#
# plt.close(1)
# fig = plt.figure(num=1)
# ax = fig.add_subplot(111, projection=ccrs.Mollweide())
# ax.contourf(lons_x[..., 0], lats_x[..., 0], dVs[..., 0], transform=ccrs.PlateCarree())
# ax.coastlines()
# fig.show()
# #