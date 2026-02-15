#!/usr/bin/env python3
"""
Test script to compare different interpolation kernels for seismic models.
"""

import numpy as np
import matplotlib.pyplot as plt
import gdrift
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def test_kernels():
    """Test different interpolation kernels on a small region."""

    print("Loading LLNL-G3Dv3 model...")
    llnl = gdrift.SeismicModel("LLNL-G3Dv3", nearest_neighbours=20)

    # Define a small test region (Pacific)
    lons = np.linspace(170, 190, 41)  # 20° longitude range
    lats = np.linspace(-10, 10, 41)   # 20° latitude range
    depth_km = 100  # 100 km depth

    # Create coordinate grid
    lons_mesh, lats_mesh = np.meshgrid(lons, lats, indexing='xy')
    depths_mesh = np.full_like(lons_mesh, depth_km)

    # Convert to Cartesian coordinates
    coordinates = gdrift.geodetic_to_cartesian(
        lats_mesh.flatten(),
        lons_mesh.flatten(),
        depths_mesh.flatten()
    )

    # Test different kernels
    kernels = {
        'IDW (Original)': {'kernel': 'idw'},
        'Gaussian (Adaptive)': {'kernel': 'gaussian'},
        'Gaussian (25km)': {'kernel': 'gaussian', 'sigma': 25000},
        'IDW Power=3': {'kernel': 'idw_power', 'power': 3.0},
        'Exponential (30km)': {'kernel': 'exponential', 'decay_length': 30000},
        'Wendland (75km)': {'kernel': 'wendland', 'support_radius': 75000}
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    for i, (name, params) in enumerate(kernels.items()):
        print(f"Testing {name}...")

        try:
            # Get VP data with specific kernel
            vp_data = llnl.at(label="dvp", coordinates=coordinates, **params)
            vp_data = vp_data.reshape(lats_mesh.shape)

            # Plot
            ax = axes[i]
            ax.set_extent([170, 190, -10, 10], crs=ccrs.PlateCarree())

            # Calculate symmetric color limits
            vmax = np.nanmax(np.abs(vp_data))
            vmin = -vmax

            # Plot data
            im = ax.contourf(lons_mesh, lats_mesh, vp_data,
                           levels=50, transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', vmin=vmin, vmax=vmax)

            # Add features
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.gridlines(draw_labels=True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                              pad=0.05, shrink=0.8)
            cbar.set_label('dVp (%)', fontsize=10)

            # Set title
            ax.set_title(f'{name}\n(Range: {vmin:.2f} to {vmax:.2f}%)',
                        fontsize=12, fontweight='bold')

            print(f"  - Range: {vmin:.2f} to {vmax:.2f}%")

        except Exception as e:
            print(f"  - Error: {e}")
            ax.text(0.5, 0.5, f'Error:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nKernel comparison plot saved as 'kernel_comparison.png'")
    print("\nKernel characteristics:")
    print("- IDW: Sharp, preserves local extrema, can be noisy")
    print("- Gaussian: Smooth, good for continuous fields")
    print("- IDW Power>1: Sharper than standard IDW")
    print("- Exponential: Smooth decay, good for correlated fields")
    print("- Wendland: Smooth with compact support, computationally efficient")

if __name__ == "__main__":
    test_kernels()
