# Depth to Pressure and Pressure to Depth
# ==========================================
# This example shows how to convert a depth to a pressure and a pressure to a depth
# Create radius and depth arrays
from gdrift import R_earth, compute_mass, compute_gravity, compute_pressure, PreliminaryRefEarthModel
import matplotlib.pyplot as plt
import numpy

# Create radius and depth arrays
radius = numpy.linspace(0., R_earth, 1000)
# Compute depths
depths = R_earth - radius

# Load PREM
prem = PreliminaryRefEarthModel()

# Use PREM to get density, then compute mass, gravity, pressure
mass = compute_mass(radius, prem.at_depth("density", depths))
gravity = compute_gravity(radius, mass)
pressure = compute_pressure(radius, prem.at_depth("density", depths), gravity)

# Now at thisd point all the values are calculated. Let's plot

plt.close(1)
fig = plt.figure(num=1)
ax = fig.add_subplot(111)
ax.plot(depths / 1e3, pressure / 1e9)
ax.set_xlabel("Depth [km]")
ax.set_ylabel("Pressure [GPa]")
fig.show()