"""
Minimal example demonstrating how to load and query a SeismicModel.
"""
import gdrift
import numpy as np
import matplotlib.pyplot as plt

# Load a seismic tomography model
model = gdrift.SeismicModel("S40RTS")

# Print available seismic models
print("Available seismic models:", gdrift.AVAILABLE_SEISMIC_MODELS)

# Query the model at some points
depths = np.linspace(100e3, 2800e3, 50)
lats = np.full_like(depths, 0.0)
lons = np.full_like(depths, 0.0)

# Convert geodetic to cartesian for querying
x, y, z = gdrift.geodetic_to_cartesian(lats, lons, depths)

# Query the model
dvs = model.query(x, y, z)

# Plot
plt.close(1)
fig = plt.figure(num=1)
ax = fig.add_subplot(111)
ax.plot(dvs, depths / 1e3)
ax.invert_yaxis()
ax.set_xlabel("dVs/Vs [%]")
ax.set_ylabel("Depth [km]")
ax.set_title("S40RTS at (0N, 0E)")
ax.grid()
plt.show()
