"""Tests for the "Usage" lines of scripts/generate_data_catalog.py.

Each line is a Python call that a user copies from the data catalog. The
tests check two things:

- `usage_line` gives, for every manifest entry with a loader, a call that
  loads that entry. The call is evaluated against stand-in classes that only
  record their arguments, so no dataset is downloaded, and the arguments are
  checked with the package's own name rules.
- The generated catalog contains exactly these lines, under the heading of
  each entry, and none of the broken `gdrift.<UTILITY KEY>(...)` form.

``scripts/`` is not a package and pytest runs with ``--import-mode=importlib``,
so the script is loaded from its path. Loading it runs its
``sys.path.insert(0, <repository root>)``, which puts the checkout first on
``sys.path`` for the rest of the test session. The tests run from that
checkout anyway, so this does not change which gdrift is imported.
"""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import gdrift
from gdrift.mineralogy import COMPOSITIONS_AVAIL, MODELS_AVAIL, dataset_name
from gdrift.profile import PreliminaryRefEarthModel

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
    """Stand-in for a gdrift loader: calling it returns (name, args).

    Only positional arguments are accepted, because `usage_line` only writes
    positional calls. A line with keyword arguments therefore fails the test,
    which is intended: the catalog shows the shortest call.
    """
    return lambda *args: (name, args)


# Stand-ins for the loaders, under the attribute names that gdrift exports.
# They cover exactly the calls that usage_line produces.
FAKE_GDRIFT = SimpleNamespace(
    RadialEarthModelFromFile=_recorder("RadialEarthModelFromFile"),
    PreliminaryRefEarthModel=_recorder("PreliminaryRefEarthModel"),
    ThermodynamicModel=_recorder("ThermodynamicModel"),
    SeismicModel=_recorder("SeismicModel"),
    EarthModel3D=SimpleNamespace(from_dataset=_recorder("EarthModel3D.from_dataset")),
)

with open(MANIFEST) as f:
    DATASETS = json.load(f)["datasets"]
DATASETS_WITH_LOADER = [d for d in DATASETS if d.get("utility")]


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
        # The class has no argument and always loads this one dataset
        assert args == () and name == PreliminaryRefEarthModel.PREM_FILENAME
    elif loader == "ThermodynamicModel":
        # Each argument must be a valid value on its own, not only joined:
        # ("SLB", "21_pyroliteCFMAS") would also join to the right name
        assert args[0] in MODELS_AVAIL, args
        assert args[1] in COMPOSITIONS_AVAIL, args
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


def test_prem_loader_on_another_dataset_uses_the_generic_class():
    """PreliminaryRefEarthModel() is only shown for 1d_prem itself."""
    line = catalog.usage_line({"name": "other_1d_model", "utility": "PREM"})
    assert line == 'gdrift.RadialEarthModelFromFile("other_1d_model")'


def test_generated_catalog_shows_the_usage_line_of_each_entry(tmp_path):
    """The catalog page lists each complete entry with its own usage line.

    This checks the place where the generator uses usage_line, so that the
    page cannot go back to printing the manifest's utility key as a class.
    """
    out = tmp_path / "catalog.md"
    catalog.DatasetCatalogGenerator(MANIFEST).generate_complete_catalog(out)
    text = out.read_text()

    # Split the page into the block of each "### <name>" entry
    blocks = dict(re.findall(r"^### (\S+)\n(.*?)(?=^### |^## |\Z)", text, flags=re.M | re.S))
    shown = [d for d in DATASETS if d["name"] in blocks]
    assert shown, "the catalog lists no entries"
    for dataset in shown:
        expected = catalog.usage_line(dataset)
        if expected:
            assert f"**Usage:** `{expected}`" in blocks[dataset["name"]], dataset["name"]

    # No line may use a UtilityClass key (all capitals) as if it were a class
    assert not re.search(r"gdrift\.[A-Z][A-Z0-9_]+\(", text)
