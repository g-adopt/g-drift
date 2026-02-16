#!/usr/bin/env python3
"""
Convert MMA-EoS output files (.out) to HDF5 format for gdrift.

This script converts thermodynamic model datasets from text-based .out files
to structured HDF5 format with proper hierarchical organization (separate /prop
and /opti groups for thermoelastic properties vs mineral phase compositions).
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata


# Bulk compositions from Chust et al. 2017 Table 2
BULK_COMPOSITIONS = {
    ('pyrolite', 'MS'): {'MgO': 60.14, 'SiO2': 39.86},
    ('pyrolite', 'FMS'): {'MgO': 53.52, 'FeO': 6.62, 'SiO2': 39.86},
    ('pyrolite', 'FMAS'): {'MgO': 52.52, 'FeO': 6.51, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('pyrolite', 'CFMS'): {'MgO': 50.85, 'FeO': 6.29, 'CaO': 3.00, 'SiO2': 39.86},
    ('pyrolite', 'CFMAS'): {'MgO': 49.91, 'FeO': 6.18, 'CaO': 2.94, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('pyrolite', 'NCFMAS'): {'MgO': 49.85, 'FeO': 6.17, 'CaO': 2.94, 'Al2O3': 2.22, 'Na2O': 0.11, 'SiO2': 38.71},
    ('depleted-mantle', 'MS'): {'MgO': 60.14, 'SiO2': 39.86},
    ('depleted-mantle', 'FMS'): {'MgO': 53.52, 'FeO': 6.62, 'SiO2': 39.86},
    ('depleted-mantle', 'FMAS'): {'MgO': 52.52, 'FeO': 6.51, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('depleted-mantle', 'CFMS'): {'MgO': 50.85, 'FeO': 6.29, 'CaO': 3.00, 'SiO2': 39.86},
    ('depleted-mantle', 'CFMAS'): {'MgO': 49.91, 'FeO': 6.18, 'CaO': 2.94, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('bulk-oceanic-crust', 'MS'): {'MgO': 60.14, 'SiO2': 39.86},
    ('bulk-oceanic-crust', 'FMS'): {'MgO': 53.52, 'FeO': 6.62, 'SiO2': 39.86},
    ('bulk-oceanic-crust', 'FMAS'): {'MgO': 52.52, 'FeO': 6.51, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('bulk-oceanic-crust', 'CFMS'): {'MgO': 50.85, 'FeO': 6.29, 'CaO': 3.00, 'SiO2': 39.86},
    ('bulk-oceanic-crust', 'CFMAS'): {'MgO': 49.91, 'FeO': 6.18, 'CaO': 2.94, 'Al2O3': 2.22, 'SiO2': 38.75},
    ('bulk-oceanic-crust', 'NCFMAS'): {'MgO': 49.85, 'FeO': 6.17, 'CaO': 2.94, 'Al2O3': 2.22, 'Na2O': 0.11, 'SiO2': 38.71},
}

# SLB version metadata from THERMODYNAMIC_SLB.md
SLB_REFERENCES = {
    '08': {
        'citation': 'Xu, W.; Lithgow-Bertelloni, C.; Stixrude, L.; Ritsema, J. "The effect of bulk composition and temperature on mantle seismic structure", Earth and Planetary Science Letters, 2008, 275, 70-79.',
        'doi': '10.1016/j.epsl.2008.08.012',
        'year': 2008,
        'solution_phases': 14,
        'total_endmembers': 46,
        'landau_wrapped': 0,
    },
    '11': {
        'citation': 'Stixrude, L.; Lithgow-Bertelloni, C. "Thermodynamics of mantle minerals -- II. Phase equilibria", Geophysical Journal International, 2011, 184, 1180-1213.',
        'doi': '10.1111/j.1365-246X.2010.04890.x',
        'year': 2011,
        'solution_phases': 14,
        'total_endmembers': 47,
        'landau_wrapped': 2,
    },
    '21': {
        'citation': 'Stixrude, L.; Lithgow-Bertelloni, C. "Thermal expansivity, heat capacity and bulk modulus of the mantle", Geophysical Journal International, 2021, 228, 1119-1149.',
        'doi': '10.1093/gji/ggaa605',
        'year': 2021,
        'solution_phases': 15,
        'total_endmembers': 51,
        'landau_wrapped': 17,
    },
    '24': {
        'citation': 'Stixrude, L.; Lithgow-Bertelloni, C. "Thermodynamics of mantle minerals -- III. The role of iron", Geophysical Journal International, 2024 (in press).',
        'doi': None,
        'year': 2024,
        'solution_phases': 15,
        'total_endmembers': 74,
        'landau_wrapped': 34,
    },
}


def parse_opti_header(filepath):
    """Extract phase names from opti.out header comments."""
    phase_names = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            # Match pattern: #[2] : Forsterite
            match = re.match(r'#\[(\d+)\]\s*:\s*(.+)', line)
            if match:
                idx, name = int(match.group(1)), match.group(2).strip()
                phase_names.append((idx, name))

    # Sort by index, return names only (indices start at 2 after P, T)
    phase_names.sort(key=lambda x: x[0])
    return [name for _, name in phase_names]


def parse_prop_file(filepath):
    """Parse prop.out → dict of property arrays."""
    column_names = ['P', 'T', 'rho', 'V', 'beta', 'alpha',
                    'kappa', 'mu', 'vp', 'vs', 'Cp', 'Cv', 'gamma']

    # Read file manually to handle malformed rows with error messages
    data_rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 13:
                # Skip malformed rows (usually containing error messages)
                continue
            # Try to convert to floats, replace failures with NaN
            row = []
            for part in parts:
                try:
                    row.append(float(part))
                except ValueError:
                    row.append(np.nan)
            data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No valid data rows in {filepath.name}")

    data = np.array(data_rows)

    # Handle case where file has only one data row (returns 1D array)
    if data.ndim == 1 or len(data) == 1:
        raise ValueError(f"Insufficient data in {filepath.name}: only {len(data_rows)} row(s) found")

    return {name: data[:, i] for i, name in enumerate(column_names)}


def parse_opti_file(filepath, phase_names):
    """Parse opti.out → dict of phase molar fraction arrays."""
    data = np.loadtxt(filepath, comments='#')
    result = {
        'P': data[:, 0],
        'T': data[:, 1],
    }
    # Phases start at column 2
    for i, name in enumerate(phase_names):
        result[name] = data[:, i + 2]
    return result


def reshape_to_grid(pressures, temperatures, values):
    """
    Convert unsorted (P, T, value) triplets to regular 2D grid.

    Returns:
        P_grid: 1D sorted unique pressures
        T_grid: 1D sorted unique temperatures
        V_grid: 2D array (len(P_grid), len(T_grid))
    """
    # Create structured array for stable sorting
    dtype = [('P', float), ('T', float), ('V', float)]
    structured = np.array(list(zip(pressures, temperatures, values)), dtype=dtype)
    structured.sort(order=['P', 'T'])

    # Extract unique axes
    P_unique = np.unique(structured['P'])
    T_unique = np.unique(structured['T'])

    # Reshape to grid
    expected = len(P_unique) * len(T_unique)
    if len(structured) == expected:
        V_grid = structured['V'].reshape(len(P_unique), len(T_unique))
    else:
        # Sparse data — a few P-T points are entirely missing from source
        print(f"    Note: {expected - len(structured)} missing P-T points, using sparse mapping")
        V_grid = np.full((len(P_unique), len(T_unique)), np.nan)
        P_to_idx = {p: i for i, p in enumerate(P_unique)}
        T_to_idx = {t: i for i, t in enumerate(T_unique)}
        for row in structured:
            i = P_to_idx.get(row['P'])
            j = T_to_idx.get(row['T'])
            if i is not None and j is not None:
                V_grid[i, j] = row['V']

    return P_unique, T_unique, V_grid


def interpolate_nan_values(pressures, temperatures, values):
    """
    Interpolate NaN values in a property array using linear griddata.

    Uses scipy.interpolate.griddata with linear method and extrapolation
    to fill missing values. This is applied to thermoelastic properties
    which vary smoothly with P and T.

    Args:
        pressures: 1D array of all pressure values (may contain NaN)
        temperatures: 1D array of all temperature values (may contain NaN)
        values: 1D array of property values (may contain NaN)

    Returns:
        filled_values: 1D array with NaN values filled by interpolation
        was_interpolated: 1D boolean array, True where values were interpolated
    """
    # Identify valid (finite) points — treats both NaN and Inf as invalid
    valid_mask = np.isfinite(values)
    n_valid = valid_mask.sum()
    n_total = len(values)

    # If less than 4 valid points, can't interpolate
    if n_valid < 4:
        return values, np.zeros(len(values), dtype=bool)

    # If all valid, nothing to do
    if n_valid == n_total:
        return values, np.zeros(len(values), dtype=bool)

    # Extract valid points
    valid_P = pressures[valid_mask]
    valid_T = temperatures[valid_mask]
    valid_vals = values[valid_mask]

    # Interpolate at all points (including valid ones, for consistency)
    # method='linear' uses Delaunay triangulation
    # fill_value=np.nan would leave extrapolated points as NaN
    # We use fill_value with nearest-neighbor extrapolation instead
    filled = griddata(
        points=np.column_stack([valid_P, valid_T]),
        values=valid_vals,
        xi=np.column_stack([pressures, temperatures]),
        method='linear',
        fill_value=np.nan  # First pass: linear only
    )

    # For points still NaN (outside convex hull), use nearest neighbor extrapolation
    still_nan = np.isnan(filled)
    if still_nan.any():
        extrapolated = griddata(
            points=np.column_stack([valid_P, valid_T]),
            values=valid_vals,
            xi=np.column_stack([pressures[still_nan], temperatures[still_nan]]),
            method='nearest'  # Extrapolate with nearest neighbor
        )
        filled[still_nan] = extrapolated

    # Track which values were interpolated (originally NaN)
    was_interpolated = ~valid_mask

    return filled, was_interpolated


def reshape_sparse_to_full_grid(pressures, temperatures, values, P_grid, T_grid):
    """
    Map sparse (P, T, value) data onto a full regular grid.

    This is used for opti.out data which may have fewer rows than prop.out
    due to LP solver failures. Missing values are filled with NaN.

    Args:
        pressures: 1D array of pressure values (may be sparse)
        temperatures: 1D array of temperature values (may be sparse)
        values: 1D array of values at those P,T points
        P_grid: Full 1D array of target pressure grid points
        T_grid: Full 1D array of target temperature grid points

    Returns:
        V_grid: 2D array (len(P_grid), len(T_grid)) with NaN for missing values
    """
    # Create full grid filled with NaN
    V_grid = np.full((len(P_grid), len(T_grid)), np.nan)

    # Create lookup dictionaries for indices
    P_to_idx = {p: i for i, p in enumerate(P_grid)}
    T_to_idx = {t: i for i, t in enumerate(T_grid)}

    # Fill in available values
    for p, t, v in zip(pressures, temperatures, values):
        i = P_to_idx.get(p)
        j = T_to_idx.get(t)
        if i is not None and j is not None:
            V_grid[i, j] = v

    return V_grid


def build_metadata(slb_version, composition, system, prop_stats, opti_stats):
    """Construct file-level metadata dictionary."""
    ref = SLB_REFERENCES[slb_version]
    bulk_comp = BULK_COMPOSITIONS.get((composition, system.upper()), {})

    return {
        'title': f'Thermodynamic properties for {composition} ({system.upper()})',
        'slb_version': slb_version,
        'slb_year': ref['year'],
        'composition': composition,
        'chemical_system': system.upper(),
        'bulk_composition_mol_pct': json.dumps(bulk_comp),
        'model_reference': ref['citation'],
        'doi': ref['doi'] if ref['doi'] else '',
        'software_reference': 'Chust, TC; Steinle-Neumann, Gerd; Dolejs, David; Schuberth, BSA; Bunge, H-P. MMA-EoS: A computational framework for mineralogical thermodynamics. Journal of Geophysical Research: Solid Earth, 2017, 122(12), 9881-9920.',
        'landau_wrapping_count': ref['landau_wrapped'],
        'solution_phases_count': ref['solution_phases'],
        'total_endmembers_count': ref['total_endmembers'],
        'author': 'Sia Ghelichkhan',
        'author_email': 'sia@gadopt.org',
        'generation_script': 'convert_slb_to_hdf5.py',
        'generation_timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'units': json.dumps({
            'Pressures': 'Pa', 'Temperatures': 'K', 'rho': 'kg/m^3',
            'V': 'm^3/mol', 'beta': '1/Pa', 'alpha': '1/K',
            'bulk_mod': 'Pa', 'shear_mod': 'Pa', 'vp': 'm/s', 'vs': 'm/s',
            'Cp': 'J/mol/K', 'Cv': 'J/mol/K', 'gamma': 'dimensionless',
            'phase_fractions': 'molar fraction (dimensionless)'
        }),
        'interpolation_prop': json.dumps({
            'method': 'scipy.interpolate.griddata with linear interpolation and nearest-neighbor extrapolation',
            'applied': True,
            'reason': 'MMA-EoS LP solver failed to find stable assemblage at some P-T conditions',
            'valid_points': prop_stats['valid'],
            'interpolated_points': prop_stats['interpolated'],
            'total_points': prop_stats['total'],
            'interpolated_fraction': prop_stats['interpolated'] / prop_stats['total'],
        }),
        'interpolation_opti': json.dumps({
            'method': 'none - phase fractions kept sparse',
            'applied': False,
            'reason': 'Phase assemblages can change discontinuously; interpolation may be non-physical',
            'valid_points': opti_stats['valid'],
            'nan_points': opti_stats['nan'],
            'total_points': opti_stats['total'],
            'valid_fraction': opti_stats['valid'] / opti_stats['total'],
        }),
    }


def write_hdf5(output_path, prop_data, opti_data, metadata):
    """
    Write grouped HDF5 file.

    Structure:
        /prop/Pressures, /prop/Temperatures, /prop/rho, ...
        /opti/Pressures, /opti/Temperatures, /opti/Forsterite, ...
    """
    with h5py.File(output_path, 'w') as f:
        # Create /prop group
        prop_grp = f.create_group('prop')
        for key, data in prop_data.items():
            prop_grp.create_dataset(key, data=data, compression='gzip', compression_opts=4)

        # Create /opti group
        opti_grp = f.create_group('opti')
        for key, data in opti_data.items():
            opti_grp.create_dataset(key, data=data, compression='gzip', compression_opts=4)

        # Write file-level metadata
        for key, value in metadata.items():
            f.attrs[key] = value


def convert_one_model(source_dir, output_dir, slb_version, composition, system):
    """Convert one SLB model from .out files to HDF5."""
    print(f"Converting SLB{slb_version} {composition} {system}...")

    # Parse source files
    prop_file = source_dir / 'prop.out'
    opti_file = source_dir / 'opti.out'

    phase_names = parse_opti_header(opti_file)
    prop_raw = parse_prop_file(prop_file)
    opti_raw = parse_opti_file(opti_file, phase_names)

    # Interpolate NaN values in prop data BEFORE reshaping
    print(f"  Interpolating property tables...")
    prop_interpolated = {}
    interpolation_masks = {}

    for key in ['rho', 'V', 'beta', 'alpha', 'vp', 'vs', 'Cp', 'Cv', 'gamma', 'kappa', 'mu']:
        filled, was_interp = interpolate_nan_values(
            prop_raw['P'],
            prop_raw['T'],
            prop_raw[key]
        )
        prop_interpolated[key] = filled
        interpolation_masks[key] = was_interp

    # Compute statistics for metadata
    total_points = len(prop_raw['P'])
    # Use rho as representative for counting interpolated points
    n_interpolated = interpolation_masks['rho'].sum()
    n_valid_original = total_points - n_interpolated

    prop_stats = {
        'total': total_points,
        'valid': int(n_valid_original),
        'interpolated': int(n_interpolated),
    }

    print(f"    Valid: {n_valid_original}/{total_points} ({100*n_valid_original/total_points:.1f}%)")
    print(f"    Interpolated: {n_interpolated}/{total_points} ({100*n_interpolated/total_points:.1f}%)")

    # Reshape to grids
    P_grid, T_grid, _ = reshape_to_grid(prop_raw['P'], prop_raw['T'], prop_interpolated['rho'])

    # Build prop group data with interpolated values
    prop_data = {
        'Pressures': P_grid,
        'Temperatures': T_grid,
    }

    for key in ['rho', 'V', 'beta', 'alpha', 'vp', 'vs', 'Cp', 'Cv', 'gamma']:
        _, _, grid = reshape_to_grid(prop_raw['P'], prop_raw['T'], prop_interpolated[key])
        prop_data[key] = grid

    # Rename kappa → bulk_mod, mu → shear_mod
    _, _, bulk_mod_grid = reshape_to_grid(prop_raw['P'], prop_raw['T'], prop_interpolated['kappa'])
    _, _, shear_mod_grid = reshape_to_grid(prop_raw['P'], prop_raw['T'], prop_interpolated['mu'])

    # Fix negative shear modulus values (non-physical MMA-EoS artifact near phase boundaries)
    neg_mask = shear_mod_grid < 0
    if neg_mask.any():
        n_neg = neg_mask.sum()
        print(f"    Fixing {n_neg} negative shear_mod values")
        P_mesh, T_mesh = np.meshgrid(P_grid, T_grid, indexing='ij')
        shear_flat = shear_mod_grid.flatten()
        P_flat = P_mesh.flatten()
        T_flat = T_mesh.flatten()
        shear_flat[neg_mask.flatten()] = np.nan
        shear_filled, _ = interpolate_nan_values(P_flat, T_flat, shear_flat)
        shear_mod_grid = shear_filled.reshape(shear_mod_grid.shape)

    prop_data['bulk_mod'] = bulk_mod_grid
    prop_data['shear_mod'] = shear_mod_grid

    # Post-reshape: interpolate any remaining NaN in grids (from missing P-T points in source)
    P_mesh, T_mesh = np.meshgrid(P_grid, T_grid, indexing='ij')
    P_flat = P_mesh.flatten()
    T_flat = T_mesh.flatten()
    for key in ['rho', 'V', 'beta', 'alpha', 'vp', 'vs', 'Cp', 'Cv', 'gamma', 'bulk_mod', 'shear_mod']:
        grid = prop_data[key]
        if np.isnan(grid).any():
            n_nan = np.isnan(grid).sum()
            print(f"    Filling {n_nan} remaining NaN in {key} (missing P-T points)")
            flat = grid.flatten()
            filled, _ = interpolate_nan_values(P_flat, T_flat, flat)
            prop_data[key] = filled.reshape(grid.shape)

    # Add interpolation mask (True where interpolated)
    _, _, interp_mask_grid = reshape_to_grid(prop_raw['P'], prop_raw['T'],
                                              interpolation_masks['rho'].astype(float))
    prop_data['interpolated_mask'] = interp_mask_grid.astype(bool)

    # Build opti group data (use sparse mapping - keep NaN where invalid)
    opti_data = {
        'Pressures': P_grid,
        'Temperatures': T_grid,
    }

    # Count opti statistics
    n_opti_valid = len(opti_raw['P'])
    n_opti_total = len(P_grid) * len(T_grid)
    opti_stats = {
        'total': n_opti_total,
        'valid': n_opti_valid,
        'nan': n_opti_total - n_opti_valid,
    }

    for phase in phase_names:
        grid = reshape_sparse_to_full_grid(opti_raw['P'], opti_raw['T'], opti_raw[phase], P_grid, T_grid)
        opti_data[phase] = grid

    # Build metadata with statistics
    metadata = build_metadata(slb_version, composition, system, prop_stats, opti_stats)

    # Write HDF5
    output_file = output_dir / f"SLB_{slb_version}_{composition}{system.upper()}.h5"
    write_hdf5(output_file, prop_data, opti_data, metadata)

    print(f"  → {output_file.name}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert MMA-EoS output files to HDF5 format'
    )
    parser.add_argument('--version', required=True, choices=['08', '11', '21', '24'],
                        help='SLB version (08, 11, 21, or 24)')
    parser.add_argument('--composition', required=True,
                        help='Composition name (e.g., pyrolite, depleted-mantle, bulk-oceanic-crust)')
    parser.add_argument('--system', required=True,
                        help='Chemical system (e.g., ms, fms, fmas, cfms, cfmas, ncfmas)')
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Directory containing prop.out and opti.out')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for HDF5 file')

    args = parser.parse_args()

    # Validate source files exist
    prop_file = args.source_dir / 'prop.out'
    opti_file = args.source_dir / 'opti.out'

    if not prop_file.exists():
        raise FileNotFoundError(f"prop.out not found in {args.source_dir}")
    if not opti_file.exists():
        raise FileNotFoundError(f"opti.out not found in {args.source_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_one_model(
        args.source_dir,
        args.output_dir,
        args.version,
        args.composition,
        args.system
    )


if __name__ == '__main__':
    main()
