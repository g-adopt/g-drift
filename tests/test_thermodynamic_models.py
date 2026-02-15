"""
Dynamic test suite for all ThermodynamicModel datasets in the registry.

Every thermodynamic model in datasets.json is downloaded and tested for:
- Loading and instantiation
- Property queries at mid-range conditions
- Convenience wrappers (vs, vp, rho)
- Inverse mapping consistency
- Out-of-bounds behaviour
"""

import pytest
import numpy as np

import gdrift
from gdrift.datasetnames import DATASET_REGISTRY, DatasetType


# Physical bounds for validation (approximate mantle ranges)
MANTLE_BOUNDS = {
    'rho': (3300, 5600),         # kg/m^3
    'bulk_mod': (100e9, 650e9),  # Pa
    'shear_mod': (50e9, 300e9),  # Pa
    'vs': (4000, 8000),          # m/s
    'vp': (8000, 14000),         # m/s
    'v_s': (4000, 8000),         # m/s (alternate naming)
    'v_p': (8000, 14000),        # m/s (alternate naming)
}


def _all_thermodynamic_names():
    """Return all thermodynamic model dataset names from the registry."""
    datasets = DATASET_REGISTRY.filter_by_type(DatasetType.THERMODYNAMIC_MODEL)
    return [ds.name for ds in datasets]


def _parse_model_name(dataset_name):
    """Parse 'SLB_XX_compositionSYSTEM' into (model_key, composition_key)."""
    parts = dataset_name.split('_')
    version = parts[1]
    comp_system = '_'.join(parts[2:])

    split_idx = None
    for i in range(1, len(comp_system)):
        if comp_system[i].isupper() and comp_system[i - 1].islower():
            split_idx = i
            break

    if split_idx:
        composition = comp_system[:split_idx]
        system = comp_system[split_idx:]
    else:
        composition = comp_system
        system = ""

    return f"SLB_{version}", f"{composition}{system}"


ALL_MODELS = _all_thermodynamic_names()


# ============================================================================
# Module-scoped fixture: load every model once
# ============================================================================

@pytest.fixture(scope="module")
def loaded_models():
    """Load all thermodynamic models once per module.

    Returns dict: {dataset_name: ThermodynamicModel}.
    Models that fail to load are stored as the exception.
    """
    results = {}
    for name in ALL_MODELS:
        model_key, composition_key = _parse_model_name(name)
        try:
            results[name] = gdrift.ThermodynamicModel(model_key, composition_key)
        except Exception as e:
            results[name] = e
    return results


# ============================================================================
# Registry Tests
# ============================================================================

def test_registry_has_thermodynamic_models():
    """Registry contains at least one thermodynamic model."""
    assert len(ALL_MODELS) > 0, "No thermodynamic models found in registry"


def test_hash_based_naming():
    """All thermodynamic models use hash-based filenames (64 hex chars + .h5)."""
    for name in ALL_MODELS:
        dataset = DATASET_REGISTRY.get_dataset(name)
        filename = dataset.get_filename()
        assert len(filename) == 67, f"{name}: invalid filename length: {filename}"
        assert filename.endswith('.h5'), f"{name}: invalid extension: {filename}"
        assert all(c in '0123456789abcdef' for c in filename[:-3]), \
            f"{name}: filename not hex: {filename}"


# ============================================================================
# Loading Tests (parametrized over every model)
# ============================================================================

@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_loads(model_name, loaded_models):
    """Each registered thermodynamic model can be loaded."""
    result = loaded_models[model_name]
    if isinstance(result, Exception):
        pytest.fail(f"Failed to load: {result}")

    assert result._pressures is not None
    assert result._depths is not None
    assert len(result._tables) > 0


# ============================================================================
# Property Tests (parametrized over every model)
# ============================================================================

@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_properties_finite(model_name, loaded_models):
    """All properties return finite values at mid-range conditions."""
    result = loaded_models[model_name]
    if isinstance(result, Exception):
        pytest.skip(f"Model failed to load: {result}")

    model = result
    test_depth = (model._depths.min() + model._depths.max()) / 2
    test_temp = 1600.0

    for prop_name in model.available_tables():
        if prop_name in {'Pressures', 'Temperatures'}:
            continue

        value = model.temperature_to_property(prop_name, test_temp, test_depth)
        assert np.isfinite(value), \
            f"{prop_name}: NaN/Inf at T={test_temp}, depth={test_depth}"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_density_in_range(model_name, loaded_models):
    """Density at mid-range conditions falls within mantle bounds."""
    result = loaded_models[model_name]
    if isinstance(result, Exception):
        pytest.skip(f"Model failed to load: {result}")

    model = result
    test_depth = (model._depths.min() + model._depths.max()) / 2
    rho = model.temperature_to_rho(1600.0, test_depth)

    assert np.isfinite(rho), "Density is NaN/Inf"
    lo, hi = MANTLE_BOUNDS['rho']
    assert lo < rho < hi, f"Density {rho:.0f} kg/m^3 outside [{lo}, {hi}]"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_convenience_wrappers(model_name, loaded_models):
    """Convenience methods (vs, vp, rho) return physical values."""
    result = loaded_models[model_name]
    if isinstance(result, Exception):
        pytest.skip(f"Model failed to load: {result}")

    model = result
    test_depth = (model._depths.min() + model._depths.max()) / 2
    test_temp = 1600.0

    tables = model.available_tables()

    if 'vs' in tables or 'v_s' in tables:
        vs = model.temperature_to_vs(test_temp, test_depth)
        assert np.isfinite(vs), "Vs is NaN/Inf"
        lo, hi = MANTLE_BOUNDS['vs']
        assert lo < vs < hi, f"Vs {vs:.0f} m/s outside [{lo}, {hi}]"

    if 'vp' in tables or 'v_p' in tables:
        vp = model.temperature_to_vp(test_temp, test_depth)
        assert np.isfinite(vp), "Vp is NaN/Inf"
        lo, hi = MANTLE_BOUNDS['vp']
        assert lo < vp < hi, f"Vp {vp:.0f} m/s outside [{lo}, {hi}]"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_inverse_mapping_consistency(model_name, loaded_models):
    """vs_to_temperature() round-trips with temperature_to_vs() within 5%."""
    result = loaded_models[model_name]
    if isinstance(result, Exception):
        pytest.skip(f"Model failed to load: {result}")

    model = result
    tables = model.available_tables()

    if 'vs' not in tables and 'v_s' not in tables:
        pytest.skip("No Vs table available")

    test_depth = (model._depths.min() + model._depths.max()) / 2
    T_original = 1600.0
    vs = model.temperature_to_vs(T_original, test_depth)
    T_recovered = model.vs_to_temperature(vs, test_depth)

    rel_error = abs(T_recovered - T_original) / T_original
    assert rel_error < 0.05, \
        f"Round-trip error {rel_error * 100:.2f}% (T={T_original} -> Vs={vs:.1f} -> T={T_recovered:.1f})"


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_invalid_property_name(loaded_models):
    """Invalid property names raise KeyError."""
    for name, result in loaded_models.items():
        if isinstance(result, Exception):
            continue
        with pytest.raises(KeyError):
            result.temperature_to_property("nonexistent_property", 1600.0, 1000e3)
        break  # one model is enough


@pytest.mark.parametrize("model_name", ALL_MODELS[:3])
def test_out_of_bounds_depth(model_name):
    """Out-of-bounds depths return NaN when extrapolate=False."""
    model_key, composition_key = _parse_model_name(model_name)
    model = gdrift.ThermodynamicModel(model_key, composition_key, extrapolate=False)

    result = model.temperature_to_rho(1600.0, 10000e3)
    assert not np.isfinite(result), f"Expected NaN for 10000 km depth, got {result}"


@pytest.mark.parametrize("model_name", ALL_MODELS[:3])
def test_out_of_bounds_temperature(model_name):
    """Out-of-bounds temperatures return NaN when extrapolate=False."""
    model_key, composition_key = _parse_model_name(model_name)
    model = gdrift.ThermodynamicModel(model_key, composition_key, extrapolate=False)

    test_depth = (model._depths.min() + model._depths.max()) / 2
    result = model.temperature_to_rho(10000.0, test_depth)
    assert not np.isfinite(result), f"Expected NaN for 10000 K, got {result}"
