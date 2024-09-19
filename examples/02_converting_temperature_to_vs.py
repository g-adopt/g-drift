import matplotlib.pyplot as plt
import numpy as np
import gdrift

# Number of random temperatures to generate
num_temperatures = 1000

# Generate random temperatures between 300 and 400
temperatures = np.random.uniform(500, 3500, num_temperatures)
depths = (np.ones_like(temperatures) *
          np.random.uniform(300e3, 1500e3, num_temperatures))

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")
seismic_speeds_s = slb_pyrolite.temperature_to_vs(
    temperature=temperatures, depth=depths)

# plotting the thermodynamic table
cntr_lines = np.linspace(4000, 7000, 20)

#
plt.close("all")
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_position([0.1, 0.1, 0.7, 0.8])

# Getting the coordinates
depths_x, temperatures_x = np.meshgrid(
    slb_pyrolite.get_depths(), slb_pyrolite.get_temperatures(), indexing="ij")

# Plotting the whole table
img = ax.contourf(
    temperatures_x,
    depths_x,
    slb_pyrolite.compute_swave_speed().get_vals(),
    cntr_lines,
    cmap=plt.colormaps["autumn"].resampled(20),
    extend="both")


# Plotting the result of conversion from randomly generate coordinates of temperature and depth
ax.scatter(temperatures, depths, c=seismic_speeds_s, edgecolors="k",
           cmap=plt.colormaps["autumn"],
           vmin=cntr_lines.min(), vmax=cntr_lines.max())
# Adjusting axis
ax.invert_yaxis()
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Depth [m]")
ax.grid()

ax.text(0.5, 1.05, s="Table versus conversions test",
        ha="center", va="center",
        transform=ax.transAxes, bbox=dict(facecolor=(1.0, 1.0, 0.7)))

fig.colorbar(img, ax=ax, cax=fig.add_axes([0.8,
             0.1, 0.02, 0.8]), orientation="vertical", label="Shear-Wave Speed [m/s]")


plt.show()