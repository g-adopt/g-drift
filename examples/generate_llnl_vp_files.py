#!/usr/bin/env python3
"""
Generate VP files for LLNL-G3D model in the same format as S40RTS files.
This script creates NetCDF files with the same grid structure and metadata as the S40RTS files.
"""

import numpy as np
import netCDF4 as nc
import gdrift
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_llnl_vp_file(depth_km, vp_data, output_dir):
    """
    Create a NetCDF file for LLNL VP data at a specific depth.

    Parameters:
    -----------
    depth_km : int
        Depth in kilometers
    vp_data : np.ndarray
        VP data array with shape (lat, lon)
    output_dir : str or Path
        Output directory path
    """

    # Create output filename
    filename = f"LLNL_dvp_{depth_km}.grd"
    filepath = Path(output_dir) / filename

    # Create longitude and latitude arrays (same as S40RTS)
    lons = np.linspace(0, 360, 721)  # 0.5 degree spacing
    lats = np.linspace(-90, 90, 361)  # 0.5 degree spacing

    # Create NetCDF file
    with nc.Dataset(filepath, 'w', format='NETCDF4') as ncfile:

        # Create dimensions
        lon_dim = ncfile.createDimension('lon', 721)
        lat_dim = ncfile.createDimension('lat', 361)

        # Create coordinate variables
        lon_var = ncfile.createVariable('lon', 'f8', ('lon',))
        lat_var = ncfile.createVariable('lat', 'f8', ('lat',))
        z_var = ncfile.createVariable('z', 'f4', ('lat', 'lon'), fill_value=np.nan)

        # Set coordinate variable attributes
        lon_var.long_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var.actual_range = [0., 360.]

        lat_var.long_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var.actual_range = [-90., 90.]

        z_var.long_name = "z"
        z_var.actual_range = [float(np.nanmin(vp_data)), float(np.nanmax(vp_data))]

        # Set coordinate data
        lon_var[:] = lons
        lat_var[:] = lats
        z_var[:, :] = vp_data

        # Set global attributes (matching S40RTS format)
        ncfile.Conventions = "COARDS, CF-1.5"
        ncfile.title = "Data gridded with continuous surface splines in tension"
        ncfile.GMT_version = "5.2.1 (r15220) [64-bit]"
        ncfile.history = f"LLNL-G3Dv3 VP data at {depth_km} km depth converted to GMT grid format"

    print(f"Created: {filepath}")


def plot_depth_slice(depth_km, vp_data, lons, lats, output_dir):
    """
    Plot a depth slice of VP data with coastlines.

    Parameters:
    -----------
    depth_km : int
        Depth in kilometers
    vp_data : np.ndarray
        VP data array with shape (lat, lon)
    lons : np.ndarray
        Longitude array
    lats : np.ndarray
        Latitude array
    output_dir : str or Path
        Output directory path
    """

    # Create figure with cartopy projection centered on Pacific
    fig = plt.figure(num=1, figsize=(12, 8))
    ax = fig.add_subplot(111, projection=ccrs.Mollweide(central_longitude=180))

    # Create meshgrid for plotting
    lons_plot, lats_plot = np.meshgrid(lons, lats)

    # Calculate symmetric color limits for diverging colormap
    vmax = np.nanmax(np.abs(vp_data))
    vmin = -vmax

    # Plot the data with diverging colormap (red-white-blue)
    im = ax.contourf(lons_plot, lats_plot, vp_data,
                     levels=100, transform=ccrs.PlateCarree(),
                     cmap='RdBu_r', extend='both',
                     vmin=vmin, vmax=vmax)

    # Add coastlines and features
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)

    # Add gridlines
    ax.gridlines(draw_labels=False, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('dVp (%)', fontsize=12)

    # Set title
    ax.set_title(f'LLNL-G3Dv3 dVp at {depth_km} km depth', fontsize=14, fontweight='bold')

    # Save the plot
    plot_filename = f"LLNL_dvp_{depth_km}_plot.png"
    plot_filepath = Path(output_dir) / plot_filename
    fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close(1)

    print(f"Plot saved: {plot_filepath}")


def main():
    """Main function to generate all LLNL VP files."""

    print("Loading LLNL-G3Dv3 model...")
    llnl = gdrift.SeismicModel("LLNL-G3Dv3", nearest_neighbours=40)

    # Define depth range (same as S40RTS files)
    depths_km = list(range(0, 2851, 50))  # 0 to 2850 km in 50 km steps

    # Create output directory
    output_dir = Path("/Users/sghelichkhani/Workplace/people/grace-shephard/LLNL_VP_interpolated")
    output_dir.mkdir(exist_ok=True)

    # Create coordinate grids
    lons = np.linspace(0, 360, 721)  # 0.5 degree spacing
    lats = np.linspace(-90, 90, 361)  # 0.5 degree spacing

    print(f"Generating VP files for {len(depths_km)} depth levels...")

    for i, depth_km in enumerate(depths_km):
        print(f"Processing depth {depth_km} km ({i + 1}/{len(depths_km)})...")

        # Create meshgrid for this depth
        lons_mesh, lats_mesh = np.meshgrid(lons, lats, indexing='xy')
        depths_mesh = np.full_like(lons_mesh, depth_km)

        # Convert to Cartesian coordinates for gdrift
        # Note: geodetic_to_cartesian expects depth in km and returns coordinates in meters
        coordinates = gdrift.geodetic_to_cartesian(
            lats_mesh.flatten(),
            lons_mesh.flatten(),
            depths_mesh.flatten() * 1e3 # depth_km is already in km
        )
        # Get VP data from LLNL model using different interpolation kernels
        try:
            # You can now choose from different interpolation kernels:

            # Original IDW (default behavior)
            # vp_data = llnl.at(label="dvp", coordinates=coordinates)

            # Gaussian kernel with adaptive bandwidth (recommended for smoother results)
            vp_data = llnl.at(label="dvp", coordinates=coordinates, kernel='gaussian')

            # Gaussian with fixed 25km bandwidth
            # vp_data = llnl.at(label="dvp", coordinates=coordinates, kernel='gaussian', sigma=25000)

            # IDW with higher power (sharper interpolation)
            # vp_data = llnl.at(label="dvp", coordinates=coordinates, kernel='idw_power', power=3.0)

            # Exponential decay with 30km characteristic length
            # vp_data = llnl.at(label="dvp", coordinates=coordinates, kernel='exponential', decay_length=30000)

            # Wendland compactly supported kernel (smooth with finite support)
            # vp_data = llnl.at(label="dvp", coordinates=coordinates, kernel='wendland', support_radius=75000)

            vp_data = vp_data.reshape(lats_mesh.shape)

            # Plot the depth slice
            plot_depth_slice(depth_km, vp_data, lons, lats, output_dir)

            # Create the NetCDF file
            create_llnl_vp_file(depth_km, vp_data, output_dir)

        except Exception as e:
            print(f"Error processing depth {depth_km} km: {e}")
            continue

    print(f"Completed! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
