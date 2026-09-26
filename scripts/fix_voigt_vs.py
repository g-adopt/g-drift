#!/usr/bin/env python3
"""Fill the NaN layers of vs in the SEMUCB-WM1 dataset with the Voigt average.

The SEMUCB-WM1 dataset of gdrift 0.1.1 comes from the GRD collection path of
convert_seismic_models.py. Its isotropic component exists only every 50 km,
while vsh and vsv exist every 5 km above 200 km, so vs is NaN in 117 of the 174
layers. convert_grd_model now fills such layers with the Voigt average of vsh
and vsv (fill_vs_from_voigt). The GRD source directories of this model are not
available, so this script applies the same function to the 0.1.1 dataset file:

* the input must be the 0.1.1 file (its sha256 is checked);
* vs is filled point by point where it is NaN and vsh and vsv are finite, after
  fill_vs_from_voigt has checked that the existing vs is the Voigt average;
* coordinates, vsh, vsv and the finite values of vs are copied unchanged;
* the file comment gets one sentence about the fill.

Where to find the input: the 0.1.1 object `69448c4c....h5` of
s3://gadopt/g-drift/ is kept as s3://gadopt/g-drift-backup-0.1.1/69448c4c....h5
(public URL
https://gadopt.syd1.digitaloceanspaces.com/g-drift-backup-0.1.1/69448c4c43e7d96daad5d7a98c59067c797d845bf332a5016e254da76572586a.h5).

Usage (from the repository root):

    python scripts/fix_voigt_vs.py --source <0.1.1 SEMUCB-WM1 file>

The output goes to gdrift/data-sia/3d_seismic_SEMUCB-WM1.h5 and to the runtime
cache gdrift/data/<hashed name>.h5. The script prints the new sha256 for
datasets.json.
"""

import argparse
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gdrift.datasetnames import hash_name  # noqa: E402

DATASET_NAME = "3d_seismic_SEMUCB-WM1"

# sha256 of the SEMUCB-WM1 file of gdrift 0.1.1, the only accepted input.
SOURCE_SHA256 = "d1c8426bd9fffecc7fb942f7886599a966be789c3da392e834b42670ce37afaa"


def load_conversion_module():
    """Import scripts/convert_seismic_models.py (not a package) by its path."""
    spec = importlib.util.spec_from_file_location(
        "convert_seismic_models", REPO / "scripts" / "convert_seismic_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Fill vs, write the new dataset and print its sha256."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="SEMUCB-WM1 dataset file of gdrift 0.1.1")
    args = parser.parse_args()

    found = sha256(args.source)
    if found != SOURCE_SHA256:
        raise SystemExit(f"{args.source} has sha256 {found}, expected the 0.1.1 file {SOURCE_SHA256}.")

    convert = load_conversion_module()

    # Read every dataset and attribute of the input file.
    with h5py.File(args.source, "r") as f:
        data = {key: f[key][()] for key in f}
        attrs = dict(f.attrs)

    filled, n_filled, max_rel = convert.fill_vs_from_voigt(data["vs"], data["vsh"], data["vsv"])
    print(f"Voigt check: largest relative difference of the existing vs {max_rel:.2e} "
          f"(tolerance {convert.VOIGT_MATCH_TOLERANCE:.0e})")
    print(f"Filled {n_filled} points; NaN left in vs: {int(np.isnan(filled).sum())}")
    if n_filled == 0:
        raise SystemExit("Nothing filled; the output would equal the input.")
    data["vs"] = filled
    attrs["comment"] = str(attrs.get("comment", "")) + (
        ". vs at the depths without an isotropic GRD file is the Voigt average "
        "sqrt((2 vsv^2 + vsh^2)/3) of vsh and vsv (scripts/fix_voigt_vs.py)."
    )

    # Write in the same flat layout as the conversion scripts.
    output_sia = REPO / "gdrift" / "data-sia" / f"{DATASET_NAME}.h5"
    convert.write_hdf5(output_sia, data, attrs)
    output_cache = REPO / "gdrift" / "data" / f"{hash_name(DATASET_NAME)}.h5"
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_sia, output_cache)
    print(f"Wrote {output_sia}")
    print(f"Copied to {output_cache}")
    print(f"SHA256: {sha256(output_sia)}")


if __name__ == "__main__":
    main()
