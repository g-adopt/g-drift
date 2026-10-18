"""Tests for the "Usage" lines of scripts/generate_data_catalog.py.

Each line is a Python call that a user copies from the data catalog. The test
evaluates every line against stand-in classes that only record their
arguments, so no dataset is downloaded, and then checks with the package's
own name rules that the arguments load the dataset that the line belongs to.

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the script is loaded from its path.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gdrift
from gdrift.mineralogy import dataset_name

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "generate_data_catalog.py"
MANIFEST = REPO / "gdrift" / "datasets.json"


def _load_script():
    """Import scripts/generate_data_catalog.py as a module and return it."""
    spec = importlib.util.spec_from_file_location("generate_data_catalog", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load_script()


def _recorder(name):
    """Stand-in for a gdrift loader: calling it returns (name, args)."""
    return lambda *args: (name, args)


# Stand-ins for the loaders, under the attribute names that gdrift exports
FAKE_GDRIFT = SimpleNamespace(
    RadialEarthModelFromFile=_recorder("RadialEarthModelFromFile"),
    PreliminaryRefEarthModel=_recorder("PreliminaryRefEarthModel"),
    ThermodynamicModel=_recorder("ThermodynamicModel"),
    SeismicModel=_recorder("SeismicModel"),
    EarthModel3D=SimpleNamespace(from_dataset=_recorder("EarthModel3D.from_dataset")),
)

with open(MANIFEST) as f:
    DATASETS_WITH_LOADER = [d for d in json.load(f)["datasets"] if d.get("utility")]


@pytest.mark.parametrize("dataset", DATASETS_WITH_LOADER, ids=lambda d: d["name"])
def test_usage_line_loads_its_own_dataset(dataset):
    """The usage line calls a real gdrift loader with arguments that name this dataset."""
    line = catalog.usage_line(dataset)
    assert line.startswith("gdrift."), f"{dataset['name']}: no usage line"

    # The class or method in the line must exist in the real package
    path = line.split("(", 1)[0].split(".")[1:]
    target = gdrift
    for attribute in path:
        target = getattr(target, attribute)

    loader, args = eval(line, {"gdrift": FAKE_GDRIFT})
    name = dataset["name"]
    if loader == "RadialEarthModelFromFile":
        assert args == (name,)
    elif loader == "PreliminaryRefEarthModel":
        # The class always loads 1d_prem
        assert args == () and name in ("1d_prem", "test")
    elif loader == "ThermodynamicModel":
        assert dataset_name(*args) == name
    elif loader == "SeismicModel":
        assert args[0] in gdrift.AVAILABLE_SEISMIC_MODELS
        assert f"3d_seismic_{args[0]}" == name
    elif loader == "EarthModel3D.from_dataset":
        assert args[0] in gdrift.AVAILABLE_EARTH_MODELS_3D
        assert f"3d_earthmodel_{args[0]}" == name
    else:
        pytest.fail(f"unexpected loader {loader}")


def test_usage_line_is_empty_without_a_loader():
    """An entry without a known loader gets no usage line."""
    assert catalog.usage_line({"name": "x"}) == ""
    assert catalog.usage_line({"name": "x", "utility": "UNKNOWN"}) == ""
