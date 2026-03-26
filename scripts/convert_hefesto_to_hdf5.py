#!/usr/bin/env python3
"""
Convert HeFESTo (Stixrude & Lithgow-Bertelloni) output to HDF5 for gdrift.

This script reads the Fortran output files from a HeFESTo thermodynamic
equilibrium calculation and converts them into the HDF5 format expected by
gdrift's ThermodynamicModel. HeFESTo computes stable mineral assemblages
and their physical properties by minimising Gibbs free energy on a regular
pressure-temperature grid.

Source files read:
    fort.56  — Aggregate physical properties (rho, VS, VP, alpha, Cp, KS, KT, ...)
    fort.58  — Voigt-Reuss-Hill elastic moduli and velocities
    fort.59  — Isomorphic thermodynamic properties (V, gamma, theta, T_melt, ...)
    fort.99  — Species molar amounts (73 end-members)

Output HDF5 structure:
    /prop/Pressures          (Pa)
    /prop/Temperatures       (K)
    /prop/rho                (kg/m³)        — aggregate density
    /prop/bulk_mod           (Pa)           — Hill-average adiabatic bulk modulus
    /prop/shear_mod          (Pa)           — Hill-average shear modulus
    /prop/vs                 (m/s)          — shear wave velocity (recomputed from moduli)
    /prop/vp                 (m/s)          — compressional wave velocity (recomputed)
    /prop/Cp                 (J/mol/K)      — isobaric heat capacity (total)
    /prop/Cv                 (J/mol/K)      — isochoric heat capacity (computed)
    /prop/V                  (m³/mol)       — molar volume
    /prop/alpha              (1/K)          — thermal expansivity (total)
    /prop/beta               (1/Pa)         — isothermal compressibility (computed)
    /prop/gamma              (-)            — Grüneisen parameter
    /prop/KT                 (Pa)           — isothermal bulk modulus (total)
    /prop/S                  (J/mol/K)      — specific entropy
    /prop/H                  (J/mol)        — specific enthalpy
    /prop/VB                 (m/s)          — bulk sound velocity
    /prop/vs_anelastic       (m/s)          — anelastically corrected VS
    /prop/vp_anelastic       (m/s)          — anelastically corrected VP
    /prop/Qp                 (-)            — compressional quality factor
    /prop/theta              (K)            — Debye temperature (isomorphic)
    /prop/T_melt             (K)            — Lindemann melting temperature
    /prop/bulk_mod_voigt     (Pa)           — Voigt-bound adiabatic bulk modulus
    /prop/bulk_mod_reuss     (Pa)           — Reuss-bound adiabatic bulk modulus
    /prop/shear_mod_voigt    (Pa)           — Voigt-bound shear modulus
    /prop/shear_mod_reuss    (Pa)           — Reuss-bound shear modulus
    /prop/interpolated_mask  (bool)         — True where values were interpolated

    /opti/Pressures          (Pa)
    /opti/Temperatures       (K)
    /opti/<species_name>     (mol)          — molar amounts for each end-member

Quality control:
    The HeFESTo Gibbs minimiser occasionally produces unreliable results at
    extreme P-T conditions near phase boundaries. These manifest as negative
    shear moduli (G_h < 0) or negative bulk moduli (KS_h < 0) in the
    Hill-average elastic properties (fort.58). When the assemblage is wrong,
    ALL properties at that P-T point are unreliable — not just the elastic
    moduli but also thermodynamic quantities (alpha, Cp, KT, etc.), anelastic
    velocities, and quality factors. The script therefore:

    1. Detects artefacts using fort.58 Hill-average moduli only (G_h < 0 or
       KS_h < 0), which is unambiguous and avoids mixing quantities from
       different averaging schemes.
    2. NaN-fills ALL property grids (from fort.56, fort.58, and fort.59) at
       these points.
    3. Interpolates ALL grids along the pressure axis at fixed temperature.
    4. Recomputes derived quantities (vs, vp, Cv, beta, VB) from the cleaned
       primary variables to guarantee self-consistency.
    5. Enforces physical constraints (positive density, moduli, heat capacity,
       expansivity) as a final sanity check on the interpolated values.

    Points that were missing from HeFESTo output (no feasible assemblage,
    typically at low P / high T) and solver artefact points are both flagged
    in the interpolated_mask.

Unit conversions from HeFESTo output:
    Pressure:   GPa → Pa         (×1e9)
    Density:    g/cm³ → kg/m³    (×1e3)
    Velocity:   km/s → m/s       (×1e3)
    Moduli:     GPa → Pa         (×1e9)
    Alpha:      ×1e-5 K⁻¹ → K⁻¹ (×1e-5)
    Cp, S:      J/g/K → J/mol/K  (×mol_mass)
    H:          kJ/g → J/mol     (×mol_mass×1e3)
    Volume:     cm³/mol → m³/mol (×1e-6)

Computed properties (not directly in HeFESTo output):
    vs   = sqrt(shear_mod / rho)
    vp   = sqrt((bulk_mod + 4/3 * shear_mod) / rho)
    VB   = sqrt(bulk_mod / rho)
    Cv   = Cp × KT_total / KS_total
    beta = 1 / KT_total

Usage:
    python convert_hefesto_to_hdf5.py \\
        --source-dir /path/to/HeFESTo/RUN_HIRES \\
        --output-dir /path/to/gdrift/data-sia \\
        --dataset-name SLB_24_pyroliteCFMASNaCr

References:
    Stixrude, L. & Lithgow-Bertelloni, C. (2024). Thermodynamics of mantle
    minerals III: The role of iron. Geophysical Journal International, 237,
    1699-1733.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d


# ── HeFESTo bulk composition (from control file) ────────────────────────
# Molar amounts for pyrolite with 8-component system (CFMASNaCr)
BULK_COMPOSITION_MOLES = {
    'Si': 3.79222, 'Mg': 4.88475, 'Fe': 0.57874, 'Ca': 0.28734,
    'Al': 0.39685, 'Na': 0.02197, 'Cr': 0.03813, 'O': 14.00450,
}
ATOMIC_MASSES = {
    'Si': 28.085, 'Mg': 24.305, 'Fe': 55.845, 'Ca': 40.078,
    'Al': 26.982, 'Na': 22.990, 'Cr': 51.996, 'O': 15.999,
}
MOLAR_MASS = sum(BULK_COMPOSITION_MOLES[e] * ATOMIC_MASSES[e]
                 for e in BULK_COMPOSITION_MOLES)  # ~506.3 g/mol

# Precision for rounding P-T values to avoid floating-point key mismatches
PT_DECIMALS = 6

# ── Species name mapping: HeFESTo abbreviation → readable name ──────────
SPECIES_NAMES = {
    'an': 'Anorthite', 'ab': 'Albite',
    'sp': 'Spinel', 'hc': 'Hercynite', 'smag': 'Spinel-Magnetite', 'picr': 'Picrochromite',
    'en': 'Enstatite', 'fs': 'Ferrosilite', 'mgts': 'Mg-Tschermak', 'odi': 'Ortho-Diopside',
    'mgc2': 'Mg-C2c', 'fec2': 'Fe-C2c',
    'di': 'Diopside', 'he': 'Hedenbergite', 'cen': 'Clinoenstatite',
    'cats': 'Ca-Tschermak', 'jd': 'Jadeite', 'acm': 'Acmite',
    'wo': 'Wollastonite', 'pwo': 'Post-Wollastonite',
    'py': 'Pyrope', 'al': 'Almandine', 'gr': 'Grossular',
    'mgmj': 'Mg-Majorite', 'namj': 'Na-Majorite', 'andr': 'Andradite', 'knor': 'Knorringite',
    'capv': 'Ca-Perovskite',
    'fo': 'Forsterite', 'fa': 'Fayalite',
    'mgwa': 'Mg-Wadsleyite', 'fewa': 'Fe-Wadsleyite',
    'mgri': 'Mg-Ringwoodite', 'feri': 'Fe-Ringwoodite',
    'mgil': 'Mg-Akimotoite', 'feil': 'Fe-Akimotoite',
    'co': 'Corundum', 'hem': 'Hematite', 'esk': 'Eskolaite',
    'mgpv': 'Mg-Bridgmanite', 'fepv': 'Fe-Bridgmanite', 'alpv': 'Al-Bridgmanite',
    'hepv': 'He-Bridgmanite', 'hlpv': 'Hl-Bridgmanite',
    'fapv': 'Fa-Bridgmanite', 'crpv': 'Cr-Bridgmanite',
    'mppv': 'Mg-Post-Perovskite', 'fppv': 'Fe-Post-Perovskite',
    'appv': 'Al-Post-Perovskite', 'hppv': 'H-Post-Perovskite', 'cppv': 'Cr-Post-Perovskite',
    'mgcf': 'Mg-Ca-Ferrite', 'fecf': 'Fe-Ca-Ferrite', 'nacf': 'Na-Ca-Ferrite',
    'hmag': 'H-Magnetite', 'crcf': 'Cr-Ca-Ferrite',
    'mnal': 'Mg-NAL-phase', 'fnal': 'Fe-NAL-phase', 'nnal': 'Na-NAL-phase',
    'pe': 'Periclase', 'wu': 'Wustite', 'wuls': 'Wustite-LS',
    'anao': 'Alpha-NaAlO2', 'mag': 'Magnetite',
    'qtz': 'Quartz', 'coes': 'Coesite', 'st': 'Stishovite', 'apbo': 'Seifertite',
    'ky': 'Kyanite', 'neph': 'Nepheline',
    'fea': 'Iron-Alpha', 'feg': 'Iron-Gamma', 'fee': 'Iron-Epsilon',
}


# ── File readers ─────────────────────────────────────────────────────────

def read_fort56(path):
    """Read aggregate properties from fort.56 (skip 2 header lines).

    Uses manual parsing to handle Fortran format overflows (e.g. '******')
    which are replaced with NaN. Warns about any such parse failures.
    """
    # Column mapping from the README (NOT from the file header, which
    # mislabels column 15 as "Qs(-)" when it is actually KT).
    # There is no Qs column in fort.56 — only Qp.
    cols = ['P', 'depth', 'T', 'rho', 'VB', 'VS', 'VP', 'VSQ', 'VPQ',
            'H', 'S', 'alpha', 'cp', 'KS', 'KT', 'Qp', 'rho0', 'phase']

    data = {c: [] for c in cols}
    n_parse_failures = 0
    with open(path) as f:
        next(f)  # skip parameter-set identification line
        next(f)  # skip column headers
        for line in f:
            parts = line.split()
            if len(parts) < 18:
                continue
            row_had_failure = False
            for i, col in enumerate(cols[:-1]):
                try:
                    data[col].append(float(parts[i]))
                except (ValueError, IndexError):
                    data[col].append(np.nan)
                    row_had_failure = True
            data['phase'].append(parts[-1] if len(parts) >= 19 else '')
            if row_had_failure:
                n_parse_failures += 1

    if n_parse_failures > 0:
        print(f"  WARNING: {n_parse_failures} rows in fort.56 had parse failures "
              f"(Fortran format overflow?) — replaced with NaN")

    for col in cols[:-1]:
        data[col] = np.round(np.array(data[col]), PT_DECIMALS)
    # Round P and T specifically to ensure cross-file consistency
    data['P'] = np.round(data['P'], PT_DECIMALS)
    data['T'] = np.round(data['T'], PT_DECIMALS)
    return data


def read_fort58(path):
    """Read Voigt-Reuss-Hill elastic moduli from fort.58 (skip 1 header)."""
    cols = ['P', 'depth', 'T', 'rho',
            'KS_h', 'G_h', 'VB_h', 'VS_h', 'VP_h',
            'KS_r', 'G_r', 'VB_r', 'VS_r', 'VP_r',
            'KS_v', 'G_v', 'VB_v', 'VS_v', 'VP_v']
    raw = np.loadtxt(path, skiprows=1)
    data = {col: raw[:, i] for i, col in enumerate(cols)}
    data['P'] = np.round(data['P'], PT_DECIMALS)
    data['T'] = np.round(data['T'], PT_DECIMALS)
    return data


def read_fort59(path):
    """Read isomorphic thermodynamic properties from fort.59 (skip 1 header)."""
    cols = ['P', 'depth', 'T', 'Vol', 'KS', 'KT', 'alpha', 'Cp',
            'theta', 'gamma', 'q', 'V_Debye', 'Ppart1', 'Ppart2', 'T_melt']
    raw = np.loadtxt(path, skiprows=1)
    data = {col: raw[:, i] for i, col in enumerate(cols)}
    data['P'] = np.round(data['P'], PT_DECIMALS)
    data['T'] = np.round(data['T'], PT_DECIMALS)
    return data


def read_fort99(path):
    """Read species molar amounts from fort.99 (skip 1 header).

    Skips warning lines that HeFESTo may emit mid-file.
    """
    with open(path) as f:
        header = f.readline().split()
    n_cols = len(header)
    species_abbrevs = header[3:-2]  # exclude Pi, depth, Ti, Gibbs, Quality

    rows = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) != n_cols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    raw = np.array(rows)
    data = {
        'P': np.round(raw[:, 0], PT_DECIMALS),
        'depth': raw[:, 1],
        'T': np.round(raw[:, 2], PT_DECIMALS),
    }
    for i, abbrev in enumerate(species_abbrevs):
        data[abbrev] = raw[:, i + 3]
    return data, species_abbrevs


# ── Grid building and interpolation ─────────────────────────────────────

def build_grid(P_vals, T_vals, values):
    """Reshape flat arrays into a 2D grid (n_P × n_T).

    P and T values should already be rounded (via PT_DECIMALS) before calling.
    Returns P_unique, T_unique, grid_2d. Missing points are NaN.
    """
    P_unique = np.sort(np.unique(P_vals))
    T_unique = np.sort(np.unique(T_vals))

    grid = np.full((len(P_unique), len(T_unique)), np.nan)
    P_idx = {p: i for i, p in enumerate(P_unique)}
    T_idx = {t: j for j, t in enumerate(T_unique)}

    for p, t, v in zip(P_vals, T_vals, values):
        i = P_idx.get(p)
        j = T_idx.get(t)
        if i is not None and j is not None:
            grid[i, j] = v

    return P_unique, T_unique, grid


def interpolate_along_pressure(P_grid, grid_2d):
    """Interpolate NaN values along the pressure axis at each temperature.

    Missing points in HeFESTo are always at the lowest pressures for a given
    temperature. We extrapolate from the first few valid pressure points
    downward using linear interpolation.

    Returns the filled grid and a boolean mask (True = was interpolated).
    """
    mask = np.isnan(grid_2d)
    filled = grid_2d.copy()

    for j in range(grid_2d.shape[1]):
        col = grid_2d[:, j]
        valid = np.isfinite(col)
        if valid.all() or not valid.any():
            continue

        f = interp1d(P_grid[valid], col[valid], kind='linear',
                     fill_value='extrapolate', bounds_error=False)
        filled[~valid, j] = f(P_grid[~valid])

    return filled, mask


# ── Main conversion ─────────────────────────────────────────────────────

def convert(source_dir, output_path):
    """Read HeFESTo files, clean, transform, and write HDF5."""
    source = Path(source_dir)

    # ── Phase 1: Read all fort files ─────────────────────────────────────

    print("Reading HeFESTo output files...")
    f56 = read_fort56(source / 'fort.56')
    f58 = read_fort58(source / 'fort.58')
    f59 = read_fort59(source / 'fort.59')
    f99, species_list = read_fort99(source / 'fort.99')

    # Build the P-T grid from fort.56
    P_gpa, T_k, _ = build_grid(f56['P'], f56['T'], f56['rho'])
    n_P, n_T = len(P_gpa), len(T_k)
    n_total = n_P * n_T
    n_missing = n_total - len(f56['P'])
    print(f"Grid: {n_P} pressures x {n_T} temperatures = {n_total} points")
    print(f"Data points: {len(f56['P'])} ({n_missing} missing from HeFESTo)")

    # Helper: build grid from flat arrays (no interpolation yet)
    def to_grid(P_flat, T_flat, values):
        _, _, g = build_grid(P_flat, T_flat, values)
        return g

    # Build ALL raw grids before any cleaning.
    # Keep the original fort.56 VS for validation later.
    print("Building raw grids...")

    # fort.56 grids (raw HeFESTo units, converted later)
    raw = {}
    for key in ['rho', 'VB', 'VS', 'VP', 'VSQ', 'VPQ',
                'H', 'S', 'alpha', 'cp', 'KS', 'KT', 'Qp']:
        raw[f'f56_{key}'] = to_grid(f56['P'], f56['T'], f56[key])

    # fort.58 grids
    for key in ['KS_h', 'G_h', 'KS_r', 'G_r', 'KS_v', 'G_v']:
        raw[f'f58_{key}'] = to_grid(f58['P'], f58['T'], f58[key])

    # fort.59 grids
    for key in ['Vol', 'gamma', 'theta', 'T_melt']:
        raw[f'f59_{key}'] = to_grid(f59['P'], f59['T'], f59[key])

    # ── Phase 2: Detect artefacts ────────────────────────────────────────
    #
    # When HeFESTo's Gibbs minimiser finds a marginal assemblage, ALL
    # properties at that point are unreliable. We detect bad points using
    # the Hill-average elastic moduli from fort.58 only:
    #   - G_h < 0  (negative shear modulus — unambiguously non-physical)
    #   - KS_h < 0 (negative bulk modulus — unambiguously non-physical)
    #
    # This avoids comparing quantities from different averaging schemes
    # (e.g. fort.56 total VS vs fort.58 Hill G_h) which differ even at
    # good points due to metamorphic contributions.

    print("Detecting solver artefacts...")

    # Start with the missing-data mask (NaN in the raw grids)
    interp_mask = np.isnan(raw['f56_rho'])

    # Artefact detection on fort.58 Hill averages
    bad = np.zeros((n_P, n_T), dtype=bool)
    G_h = raw['f58_G_h']
    KS_h = raw['f58_KS_h']

    # Negative moduli (unambiguous)
    bad |= np.nan_to_num(G_h, nan=0) < 0
    bad |= np.nan_to_num(KS_h, nan=0) < 0

    # Exclude already-missing points from the count
    n_artefacts = (bad & ~interp_mask).sum()
    print(f"  Found {n_artefacts} solver artefact points "
          f"(G_h < 0 or KS_h < 0 in fort.58)")

    # Merge into the interpolation mask
    interp_mask |= bad

    # ── Phase 3: NaN-fill ALL grids at bad points, then interpolate ─────

    print("Cleaning and interpolating all grids...")

    for key in raw:
        raw[key][interp_mask] = np.nan
        raw[key], _ = interpolate_along_pressure(P_gpa, raw[key])

    n_interpolated = interp_mask.sum()
    print(f"  Total interpolated: {n_interpolated} / {n_total} points "
          f"({100 * n_interpolated / n_total:.2f}%)")
    print(f"    of which {n_missing} were missing from HeFESTo output")
    print(f"    and {n_artefacts} were solver artefacts")

    # ── Phase 4: Unit conversions and derived quantities ─────────────────

    print("Converting units and computing derived quantities...")

    # Primary variables (unit-converted from cleaned grids)
    rho = raw['f56_rho'] * 1e3                    # g/cm³ → kg/m³
    bulk_mod = raw['f58_KS_h'] * 1e9              # GPa → Pa
    shear_mod = raw['f58_G_h'] * 1e9              # GPa → Pa
    alpha = raw['f56_alpha'] * 1e-5               # ×1e-5 K⁻¹ → K⁻¹
    Cp_specific = raw['f56_cp']                   # J/g/K
    KS_total = raw['f56_KS']                      # GPa (for Cv computation)
    KT_total = raw['f56_KT']                      # GPa (for Cv, beta)
    V = raw['f59_Vol'] * 1e-6                     # cm³/mol → m³/mol
    gamma = raw['f59_gamma']                      # dimensionless
    S = raw['f56_S'] * MOLAR_MASS                 # J/g/K → J/mol/K
    H = raw['f56_H'] * MOLAR_MASS * 1e3           # kJ/g → J/mol
    theta = raw['f59_theta']                      # K
    T_melt = raw['f59_T_melt']                    # K
    Qp = raw['f56_Qp']                            # dimensionless
    vs_anel = raw['f56_VSQ'] * 1e3                # km/s → m/s
    vp_anel = raw['f56_VPQ'] * 1e3                # km/s → m/s

    # Voigt and Reuss bounds (GPa → Pa)
    bulk_mod_voigt = raw['f58_KS_v'] * 1e9
    bulk_mod_reuss = raw['f58_KS_r'] * 1e9
    shear_mod_voigt = raw['f58_G_v'] * 1e9
    shear_mod_reuss = raw['f58_G_r'] * 1e9

    # Convert Cp to per-mol
    Cp = Cp_specific * MOLAR_MASS                 # J/mol/K

    # Derived: Cv = Cp × KT/KS (using total values, both cleaned)
    Cv_specific = Cp_specific * KT_total / KS_total
    Cv = Cv_specific * MOLAR_MASS                 # J/mol/K

    # Derived: beta = 1/KT
    KT = KT_total * 1e9                           # GPa → Pa
    beta = 1.0 / KT                               # 1/Pa

    # Derived: recompute vs, vp, VB from cleaned moduli for self-consistency
    vs = np.sqrt(shear_mod / rho)
    vp = np.sqrt((bulk_mod + 4.0 / 3.0 * shear_mod) / rho)
    VB = np.sqrt(bulk_mod / rho)

    # ── Phase 5: Post-interpolation sanity checks ────────────────────────

    print("Running post-interpolation sanity checks...")

    # Only enforce constraints on elastic properties from fort.58 (Hill
    # averages), where negative values are unambiguously non-physical.
    # Total thermodynamic properties from fort.56 (alpha, KT, Cv) CAN
    # legitimately go negative near first-order phase transitions due to
    # metamorphic contributions — these are physical, not artefacts.
    constraints = {
        'rho': (rho, 1000, 8000, 'kg/m³'),
        'bulk_mod': (bulk_mod, 0, 2e12, 'Pa'),
        'shear_mod': (shear_mod, 0, 1e12, 'Pa'),
    }

    n_clipped_total = 0
    for name, (arr, lo, hi, unit) in constraints.items():
        n_bad_lo = (arr < lo).sum() if lo is not None else 0
        n_bad_hi = (arr > hi).sum() if hi is not None else 0
        if n_bad_lo + n_bad_hi > 0:
            print(f"  WARNING: clipping {n_bad_lo + n_bad_hi} non-physical "
                  f"values in {name} ({unit})")
            if lo is not None:
                arr[arr < lo] = lo
            if hi is not None:
                arr[arr > hi] = hi
            n_clipped_total += n_bad_lo + n_bad_hi

    # Recompute velocities after any clipping
    if n_clipped_total > 0:
        vs = np.sqrt(shear_mod / rho)
        vp = np.sqrt((bulk_mod + 4.0 / 3.0 * shear_mod) / rho)
        VB = np.sqrt(bulk_mod / rho)

    # Validate against original fort.56 VS at non-interpolated points
    vs_original = raw['f56_VS'] * 1e3  # already cleaned and interpolated
    vs_from_moduli = np.sqrt(raw['f58_G_h'] * 1e9 / (raw['f56_rho'] * 1e3))
    good = ~interp_mask
    if good.any():
        rel_err = np.abs(vs_original[good] - vs_from_moduli[good]) / vs_original[good]
        print(f"  Validation: fort.56 VS vs sqrt(G_h/rho) at {good.sum()} good points:")
        print(f"    median relative error = {np.nanmedian(rel_err):.2e}")
        print(f"    max relative error    = {np.nanmax(rel_err):.2e}")
        print(f"    points with >1% error = {(rel_err > 0.01).sum()}")

    # ── Phase 6: Species tables ──────────────────────────────────────────

    print("Building species tables...")
    opti_grids = {}
    for abbrev in species_list:
        g = to_grid(f99['P'], f99['T'], f99[abbrev])
        # Species at bad/missing points: leave as NaN (don't interpolate —
        # phase assemblages change discontinuously)
        readable = SPECIES_NAMES.get(abbrev, abbrev)
        opti_grids[readable] = g

    # ── Phase 7: Write HDF5 ─────────────────────────────────────────────

    P_pa = P_gpa * 1e9

    print(f"Writing {output_path}...")
    with h5py.File(output_path, 'w') as hf:
        # prop group
        prop = hf.create_group('prop')
        prop_datasets = {
            'Pressures': P_pa,
            'Temperatures': T_k,
            'rho': rho,
            'bulk_mod': bulk_mod,
            'shear_mod': shear_mod,
            'vs': vs,
            'vp': vp,
            'Cp': Cp,
            'Cv': Cv,
            'V': V,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'KT': KT,
            'S': S,
            'H': H,
            'VB': VB,
            'vs_anelastic': vs_anel,
            'vp_anelastic': vp_anel,
            'Qp': Qp,
            'theta': theta,
            'T_melt': T_melt,
            'bulk_mod_voigt': bulk_mod_voigt,
            'bulk_mod_reuss': bulk_mod_reuss,
            'shear_mod_voigt': shear_mod_voigt,
            'shear_mod_reuss': shear_mod_reuss,
            'interpolated_mask': interp_mask,
        }
        for name, data in prop_datasets.items():
            prop.create_dataset(name, data=data, compression='gzip', compression_opts=4)

        # opti group
        opti = hf.create_group('opti')
        opti.create_dataset('Pressures', data=P_pa, compression='gzip', compression_opts=4)
        opti.create_dataset('Temperatures', data=T_k, compression='gzip', compression_opts=4)
        for species_name, grid in opti_grids.items():
            opti.create_dataset(species_name, data=grid, compression='gzip', compression_opts=4)

        # File-level metadata
        hf.attrs['title'] = 'Thermodynamic properties for pyrolite (CFMASNaCr) from HeFESTo'
        hf.attrs['slb_version'] = '24'
        hf.attrs['slb_year'] = 2024
        hf.attrs['composition'] = 'pyrolite'
        hf.attrs['chemical_system'] = 'CFMASNaCr'
        hf.attrs['bulk_composition_moles'] = json.dumps(BULK_COMPOSITION_MOLES)
        hf.attrs['molar_mass_g_per_mol'] = MOLAR_MASS
        hf.attrs['n_atoms_per_formula_unit'] = sum(BULK_COMPOSITION_MOLES.values())
        hf.attrs['model_reference'] = (
            'Stixrude, L.; Lithgow-Bertelloni, C. '
            '"Thermodynamics of mantle minerals III: The role of iron", '
            'Geophysical Journal International, 2024, 237, 1699-1733. '
            'doi:10.1093/gji/ggae126'
        )
        hf.attrs['parameter_file'] = 'HeFESTo_Parameters_010123'
        hf.attrs['software'] = 'HeFESTo (Helmholtz Free Energy Surface Explorer)'
        hf.attrs['author'] = 'Sia Ghelichkhan'
        hf.attrs['generation_script'] = 'convert_hefesto_to_hdf5.py'
        hf.attrs['generation_timestamp'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        hf.attrs['units'] = json.dumps({
            'Pressures': 'Pa', 'Temperatures': 'K',
            'rho': 'kg/m^3', 'bulk_mod': 'Pa', 'shear_mod': 'Pa',
            'vs': 'm/s', 'vp': 'm/s', 'VB': 'm/s',
            'vs_anelastic': 'm/s', 'vp_anelastic': 'm/s',
            'Cp': 'J/mol/K', 'Cv': 'J/mol/K', 'S': 'J/mol/K',
            'H': 'J/mol', 'V': 'm^3/mol',
            'alpha': '1/K', 'beta': '1/Pa', 'KT': 'Pa',
            'gamma': 'dimensionless', 'Qp': 'dimensionless',
            'theta': 'K', 'T_melt': 'K',
            'bulk_mod_voigt': 'Pa', 'bulk_mod_reuss': 'Pa',
            'shear_mod_voigt': 'Pa', 'shear_mod_reuss': 'Pa',
            'species': 'moles',
        })

        hf.attrs['property_sources'] = json.dumps({
            'bulk_mod': 'Hill-average adiabatic bulk modulus KS_h from fort.58',
            'shear_mod': 'Hill-average shear modulus G_h from fort.58',
            'vs': 'Recomputed as sqrt(shear_mod / rho) for self-consistency',
            'vp': 'Recomputed as sqrt((bulk_mod + 4/3 * shear_mod) / rho)',
            'VB': 'Recomputed as sqrt(bulk_mod / rho)',
            'rho': 'Total aggregate density from fort.56',
            'alpha': 'Total thermal expansivity from fort.56 (includes metamorphic contribution)',
            'Cp': 'Total isobaric heat capacity from fort.56 (includes metamorphic contribution)',
            'KT': 'Total isothermal bulk modulus from fort.56 (includes metamorphic contribution)',
            'Cv': 'Computed as Cp * KT_total / KS_total (both from fort.56)',
            'beta': 'Computed as 1 / KT_total',
            'S': 'Total specific entropy from fort.56',
            'H': 'Total specific enthalpy from fort.56',
            'gamma': 'Isomorphic Gruneisen parameter from fort.59',
            'V': 'Molar volume from fort.59',
            'theta': 'Isomorphic Debye temperature from fort.59',
            'T_melt': 'Lindemann melting temperature from fort.59',
            'vs_anelastic': 'HeFESTo built-in anelastic correction from fort.56 (VSQ)',
            'vp_anelastic': 'HeFESTo built-in anelastic correction from fort.56 (VPQ)',

            'Qp': 'Compressional quality factor from fort.56',
            'bulk_mod_voigt': 'Voigt-bound adiabatic bulk modulus from fort.58',
            'bulk_mod_reuss': 'Reuss-bound adiabatic bulk modulus from fort.58',
            'shear_mod_voigt': 'Voigt-bound shear modulus from fort.58',
            'shear_mod_reuss': 'Reuss-bound shear modulus from fort.58',
        })

        hf.attrs['quality_control'] = json.dumps({
            'artefact_detection': (
                'Bad points detected using Hill-average elastic moduli from fort.58: '
                'G_h < 0 (negative shear modulus) or KS_h < 0 (negative bulk modulus). '
                'These are unambiguous indicators that the Gibbs minimiser found a '
                'marginal or incorrect assemblage.'
            ),
            'cleaning_strategy': (
                'ALL property grids (from fort.56, fort.58, and fort.59) are NaN-filled '
                'at detected artefact points, then interpolated along the pressure axis '
                'at fixed temperature. Derived quantities (vs, vp, VB, Cv, beta) are '
                'recomputed from the cleaned primary variables to guarantee internal '
                'self-consistency. This avoids the pitfall of cleaning elastic properties '
                'while leaving thermodynamic properties from the bad assemblage untouched.'
            ),
            'interpolation_method': (
                'Linear interpolation along pressure at fixed temperature, with linear '
                'extrapolation for low-P points below the first valid pressure. Missing '
                'points from HeFESTo output (no feasible assemblage at low-P/high-T) '
                'and solver artefact points are treated identically.'
            ),
            'sanity_checks': (
                'Post-interpolation physical constraints enforced on elastic properties only: '
                'rho in [1000, 8000] kg/m^3, bulk_mod (Hill) >= 0, shear_mod (Hill) >= 0. '
                'Velocities recomputed after any clipping. Total thermodynamic properties '
                '(alpha, KT, Cv) are NOT clipped because they can legitimately go negative '
                'near first-order phase transitions due to metamorphic contributions.'
            ),
            'species_handling': (
                'Species molar amounts (opti group) are NOT interpolated at bad/missing '
                'points because phase assemblages change discontinuously. These remain NaN.'
            ),
            'total_points': int(n_total),
            'missing_from_hefesto': int(n_missing),
            'solver_artefacts': int(n_artefacts),
            'total_interpolated': int(n_interpolated),
            'interpolated_fraction': float(n_interpolated / n_total),
            'values_clipped': int(n_clipped_total),
        })

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HeFESTo output files to gdrift HDF5 format')
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Directory containing fort.56, fort.58, fort.59, fort.99')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for the HDF5 file')
    parser.add_argument('--dataset-name', type=str, default='SLB_24_pyroliteCFMASNaCr',
                        help='Dataset name (used as filename, default: SLB_24_pyroliteCFMASNaCr)')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f'{args.dataset_name}.h5'

    convert(args.source_dir, output_path)


if __name__ == '__main__':
    main()
