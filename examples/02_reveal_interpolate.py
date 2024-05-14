from gadopt import *
import numpy as np
from pathlib import Path
import h5py
from scipy.spatial import cKDTree

R_earth = 6371e3  # Earth's radius in meters
R_CMB = R_earth - 2890e3  # Core-mantle boundary radius in meters
R_nd_earth = 2.22  # Non-dimensional radius of Earth
R_nd_CMB = 1.22  # Non-dimensional radius of CMB


def load_reveal():
    reveal_filename = Path(__file__).parents[1] / "gdrift/data/reveal.h5"

    # Open the HDF5 file
    with h5py.File(reveal_filename, 'r') as file:
        # Read coordinates
        coordinates = file['MODEL/coordinates'][:]
        coordinates = np.mean(coordinates, axis=1)
        is_in_mantle =  np.sqrt(np.sum(coordinates ** 2, axis=1)) > R_CMB
        coordinates = coordinates[is_in_mantle]

        # Read data
        data = file['MODEL/data'][:]

        # Dimension labels to identify the index of VSH and VSV
        dimension_labels = file['MODEL/data'].attrs['DIMENSION_LABELS'][1]
        properties = dimension_labels.strip('[]').split(' | ')
        vsv_index = properties.index('VSV')
        vsh_index = properties.index('VSH')

        # Use the gll weights
        x, w = GLLqPoints_3d(3)

        # Extract VSV and VSH data
        vsv_data = data[:, vsv_index, :]
        vsh_data = data[:, vsh_index, :]

        vsv_data = np.einsum("j, ij -> i", w, vsv_data)[is_in_mantle] / np.sum(w)
        vsh_data = np.einsum("j, ij -> i", w, vsh_data)[is_in_mantle] / np.sum(w)
    return coordinates, vsv_data, vsh_data


def interpolate_to_icocaehdron(ref_level=6, nlayers=32):
   # Set up geometry:
   # Construct a CubedSphere mesh and then extrude into a sphere:
   mesh2d = CubedSphereMesh(R_nd_CMB, refinement_level=ref_level, degree=2)
   mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')

   # Function Spaces
   Q = FunctionSpace(mesh, "CG", 1)  #
   DPC = FunctionSpace(mesh, "DPC", 0)  #
   V = VectorFunctionSpace(mesh, DPC.ufl_element()) #

   vs_v_dpc = Function(DPC, name="vs_v_dpc")
   coords = Function(V, name="Coordinates").interpolate(SpatialCoordinate(mesh))

   reveal_coords, reveal_vsv, reveal_vsh = load_reveal()
   reveal_coords = nondimensionalise_coords(reveal_coords)

   tree = cKDTree(reveal_coords)

   dists, indices = tree.query(coords.dat.data, k=500)

   # Handle zero distances by setting a small epsilon
   dists[dists == 0] = 1e-10

   # Calculate weights using inverse distance weighting
   weights = 1 / dists
   weights /= np.sum(weights, axis=1, keepdims=True)

   # Interpolate vs_v_dpc
   vs_v_dpc.dat.data[:] = np.sum(weights * reveal_vsv[indices], axis=1)

   # Optionally handle any remaining zero distance cases (for exact matches)
   exact_matches = np.isclose(dists[:, 0], 0, atol=1e-10)
   vs_v_dpc.dat.data[exact_matches] = reveal_vsv[indices[exact_matches, 0]]
   vs_v = Function(Q, name="vs_v").project(vs_v_dpc)

   VTKFile("sample.pvd").write(vs_v)


def lgP(n, xi):
   """
   Evaluates P_{n}(xi) using an iterative algorithm
   """
   if n == 0:
      return np.ones(xi.size)
   elif n == 1:
      return xi
   else:
      fP, sP = np.ones(xi.size), xi.copy()
      for i in range(2, n + 1):
         fP, sP = sP, ((2 * i - 1) * xi * sP - (i - 1) * fP) / i
      return sP

def dLgP(n, xi):
   """
   Evaluates the first derivative of P_{n}(xi)
   """
   return n * (lgP(n - 1, xi) - xi * lgP(n, xi)) / (1 - xi ** 2)

def d2LgP(n, xi):
   """
   Evaluates the second derivative of P_{n}(xi)
   """
   return (2 * xi * dLgP(n, xi) - n * (n + 1) * lgP(n, xi)) / (1 - xi ** 2)

def d3LgP(n, xi):
   """
   Evaluates the third derivative of P_{n}(xi)
   """
   return (4 * xi * d2LgP(n, xi) - (n * (n + 1) - 2) * dLgP(n, xi)) / (1 - xi ** 2)

def gLLNodesAndWeights(n, epsilon=1e-15):
   """
   Computes the GLL nodes and weights
   """
   if n < 2:
      raise ValueError(f'Error: n:{n} must be larger than 1')

   x = np.empty(n)
   w = np.empty(n)
   x[0], x[n - 1] = -1, 1
   w[0] = w[n - 1] = 2.0 / (n * (n - 1))

   n_2 = n // 2
   for i in range(1, n_2 + 1):
      xi = np.cos((4 * i - 1) * np.pi / (4 * n - 1))
      error = 1.0
      while error > epsilon:
         y = dLgP(n - 1, xi)
         dx = y / (d2LgP(n - 1, xi) - y * d3LgP(n - 1, xi) / (2 * y))
         xi -= dx
         error = abs(dx)

      x[i] = xi
      x[n - i - 1] = -xi
      w[i] = w[n - i - 1] = 2 / (n * (n - 1) * lgP(n - 1, xi) ** 2)

   if n % 2 != 0:
      x[n_2] = 0
      w[n_2] = 2.0 / (n * (n - 1) * lgP(n - 1, np.array([0]))[0] ** 2)

   return x, w

def tensor_product_3d(points_1d, weights_1d):
   """Create a 3D tensor product of 1D points and weights."""
   points_3d = np.array(np.meshgrid(points_1d, points_1d, points_1d)).T.reshape(-1, 3)
   weights_3d = np.outer(np.outer(weights_1d, weights_1d), weights_1d).flatten()
   return points_3d, weights_3d


def GLLqPoints_3d(degree):
    x, w = gLLNodesAndWeights(n=3)
    x3d, w3d = tensor_product_3d(x, w)
    return x3d, w3d

def nondimensionalise_coords(reveal_coords):
    # Calculate the slope (a)
    a = (R_nd_earth - R_nd_CMB) / (R_earth - R_CMB)
    # Calculate the intercept (b)
    b = R_nd_earth - a * R_earth
    return a * reveal_coords + b


if __name__ == "__main__":
   interpolate_to_icocaehdron()