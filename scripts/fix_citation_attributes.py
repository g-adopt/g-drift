#!/usr/bin/env python3
"""Correct the citation attributes of the SLB_21, SLB_24 and adiabat datasets.

The script reads each dataset as released in gdrift 0.1.4 (the files are
pinned by their sha256 in INPUT_SHA256), changes only HDF5 root attributes,
checks that every array in the file is unchanged, and with --write saves the
new files and updates gdrift/datasets.json. A file that changes gets a new
sha256 and therefore a new file name on the server. The 0.1.4 files stay on
the server unchanged, so older releases keep working.

Fixes:

- SLB_21 (15 files). The attribute doi is 10.1093/gji/ggaa605, which is
  a different paper (Zhao et al., 2021). The DOI of Stixrude &
  Lithgow-Bertelloni (2021) is 10.1093/gji/ggab394.
- SLB_24 tables computed with MMA-EoS (9 files, CFMS/FMS/MS). The attribute
  model_reference says "2024 (in press)" and doi is empty. The paper
  is GJI 237(3), 1699-1733, doi 10.1093/gji/ggae126.
- SLB_24_pyroliteCFMASNaCr (computed with HeFESTo). model_reference
  lists a third author who is not an author of the paper and has no DOI. The
  new text is the one that scripts/convert_hefesto_to_hdf5.py writes, and
  the file gets a doi attribute like the other SLB tables.
- 1d_geodynamic_SLB21_pyroliteCFMAS. The attribute Software names the
  MMA-EoS code of Chust et al. (2017) as "EoS_MMO". The file also gets
  model_reference and doi attributes for the SLB 2021 parameters that
  the profiles are computed from.

Usage:
    python scripts/fix_citation_attributes.py            # dry run: fix in memory copies, check, report
    python scripts/fix_citation_attributes.py --write    # also write the files and the manifest

The input files are downloaded from the CDN into a temporary directory. With
--write, each fixed file is saved twice: as gdrift/data-sia/<dataset name>.h5
(the input of scripts/upload_to_s3.py) and as gdrift/data/<sha256>.h5 (where
the loader finds it before the upload). The manifest gets the new sha256 and
filename, and `etag: null` until the upload.
"""
import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

# Reuse the hashing function of the package, so that the sha256 written into
# the manifest is computed exactly as the loader computes it.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gdrift.io import file_hash  # noqa: E402

MANIFEST_PATH = REPO / "gdrift" / "datasets.json"
DATA_SIA_DIR = REPO / "gdrift" / "data-sia"
DATA_DIR = REPO / "gdrift" / "data"

# sha256 of each input file as released in gdrift 0.1.4. The script refuses
# to work on any other version of a file, so that a second run (after the
# manifest already points to the fixed files) cannot change them again.
INPUT_SHA256 = {
    "1d_geodynamic_SLB21_pyroliteCFMAS": "46fd234ce94146632a397f5e58e82f788a3d58640cabbf1c6f89f8e9baed85e2",
    "SLB_21_bulk-oceanic-crustCFMAS": "feb990a653a3c0fec008669b3c64c43f79e73467df37ef4e0ceb6c2e62aafb5e",
    "SLB_21_bulk-oceanic-crustCFMS": "56259f390ebbad9691baa95dc8c799d7f1fdd64e8162e9a30774d6944e740158",
    "SLB_21_bulk-oceanic-crustFMAS": "59f71e2c6ea83fc3600b4340da7778606f6beb00389d36061506262c1c8c0a22",
    "SLB_21_bulk-oceanic-crustFMS": "4f2e6fa5c6947948f5195865bc5fae39a16949e6c40d0856d52bece75139bade",
    "SLB_21_bulk-oceanic-crustMS": "5136a24203e3310a8c864e9f3b508ab8ea4ccda86519d33c3e6976ead0102969",
    "SLB_21_depleted-mantleCFMAS": "ede5dfa81ac4f30b65691490dd0ccc24aea42de5e14ae14d91dfaee2296c9f7c",
    "SLB_21_depleted-mantleCFMS": "2a61028077188d14baad80154bbd35e18820633105bb30ea89866cb31c9f8c16",
    "SLB_21_depleted-mantleFMAS": "01f76c6cd3efaf27cc7857a7fcabe2390127272370d3ce8c9b7793b1b0e6b879",
    "SLB_21_depleted-mantleFMS": "aa16c697da1d53989ec1a1db434b611894bd878c9e448947722b5335103764be",
    "SLB_21_depleted-mantleMS": "a4f7905458e2844abfaa29d65618aa539d972bc913fb33d0d0c8eab5ad8b3748",
    "SLB_21_pyroliteCFMAS": "7890c7dfae31dab9684247aeea5317a3b95ef5e202ef972abc7ea5b3ea509b4f",
    "SLB_21_pyroliteCFMS": "08dde0c005efd925c6cbb178ff5176f8c172b2732c2bcfbbfaba0cfb59ddb9ad",
    "SLB_21_pyroliteFMAS": "19bc79fd2b3e5621c19f91a0b2d2aa2249ffad3cb65bb68593bec46f84d297df",
    "SLB_21_pyroliteFMS": "81daa5050f2b554eddbba0c13ce8b6fd0fa95885fb80ef95beed93daf36256ed",
    "SLB_21_pyroliteMS": "892bfc5ffd9d15ee95813fc1a06def36b45b70d762b11377b6505a452a31f060",
    "SLB_24_bulk-oceanic-crustCFMS": "99bc9dfe7d5992154fadfdd3b2c265802519d8ebb0d09053d060f1d670a817f4",
    "SLB_24_bulk-oceanic-crustFMS": "bb6f88086fadc6c39dbc87eddefc0497997c3457b919596f8cbe4ab889c67401",
    "SLB_24_bulk-oceanic-crustMS": "073c48f949959df6485ba798e5cc41d3209a6d996a1544ddda368979e4457824",
    "SLB_24_depleted-mantleCFMS": "35e7b86de70887e1181b74e06b75978bdae127aed59b15eed0bdf47f29b590cc",
    "SLB_24_depleted-mantleFMS": "7d7427633832e5bed3033be0d8f13e14e79edb80bcfaa861aa208042c72d9711",
    "SLB_24_depleted-mantleMS": "474e68a95e899b0cc9d0834fb181b404843b01bbd2b7b5095a7c4883d23dd790",
    "SLB_24_pyroliteCFMS": "95b2202d5682c9b2b89101b66de588fa4e63610e6f7ae22c40b5034ea97dd271",
    "SLB_24_pyroliteFMS": "f25942689353d530fdd23b0ec605af040fe1d2296fa1f255e86973370487ce3d",
    "SLB_24_pyroliteMS": "b622169b4c3e665fc7c6d3525179a17aa765156b9e491757c9efbb8a3dc68826",
    "SLB_24_pyroliteCFMASNaCr": "b87ce8aafa14b9aa1275084dd19128b44787ea2b69c6d2e9c60f014139574991",
}

# Bibliographic text, in the format that the conversion scripts already use.
SLB21_DOI = "10.1093/gji/ggab394"
SLB21_REFERENCE = (
    'Stixrude, L.; Lithgow-Bertelloni, C. "Thermal expansivity, heat capacity and bulk '
    'modulus of the mantle", Geophysical Journal International, 2021, 228, 1119-1149.'
)
SLB24_DOI = "10.1093/gji/ggae126"
SLB24_REFERENCE = (
    'Stixrude, L.; Lithgow-Bertelloni, C. "Thermodynamics of mantle minerals - III: the '
    'role of iron", Geophysical Journal International, 2024, 237, 1699-1733.'
)
MMA_EOS_REFERENCE = (
    "MMA-EoS: Chust, T. C., Steinle-Neumann, G., Dolejs, D., Schuberth, B. S. A., & "
    "Bunge, H.-P. (2017). MMA-EoS: A computational framework for mineralogical "
    "thermodynamics. Journal of Geophysical Research: Solid Earth, 122(12), 9881-9920. "
    "doi:10.1002/2017JB014501"
)


def new_attributes(name):
    """Return the root attributes to set for one dataset.

    Args:
        name (str): Dataset name, a key of INPUT_SHA256.

    Returns:
        dict: Attribute name to new string value. Attributes that are not in
        the dict stay as they are.
    """
    if name.startswith("SLB_21_"):
        return {"doi": SLB21_DOI}
    if name == "SLB_24_pyroliteCFMASNaCr":
        # Same text as scripts/convert_hefesto_to_hdf5.py, which includes the DOI.
        return {"model_reference": f"{SLB24_REFERENCE[:-1]}. doi:{SLB24_DOI}", "doi": SLB24_DOI}
    if name.startswith("SLB_24_"):
        return {"model_reference": SLB24_REFERENCE, "doi": SLB24_DOI}
    if name == "1d_geodynamic_SLB21_pyroliteCFMAS":
        return {
            "Software": MMA_EOS_REFERENCE,
            "model_reference": SLB21_REFERENCE,
            "doi": SLB21_DOI,
        }
    raise KeyError(name)


def read_arrays(path):
    """Read every dataset (array) of an HDF5 file into memory.

    Args:
        path (Path): HDF5 file.

    Returns:
        dict: HDF5 path of each array to its numpy value.
    """
    arrays = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda key, obj: arrays.__setitem__(key, obj[()]) if isinstance(obj, h5py.Dataset) else None)
    return arrays


def same_arrays(a, b):
    """Return True if two dicts from read_arrays hold the same keys and equal arrays (NaN equals NaN)."""
    if a.keys() != b.keys():
        return False
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def fix_one(name, entry, cdn_url, workdir):
    """Download one input file, fix its attributes in a copy, and check the copy.

    Args:
        name (str): Dataset name.
        entry (dict): Manifest entry of the dataset.
        cdn_url (str): Base URL of the server.
        workdir (Path): Directory for the downloaded and fixed files.

    Returns:
        Path: The fixed file in workdir.
    """
    source = workdir / f"{name}.orig.h5"
    urllib.request.urlretrieve(cdn_url + f"{INPUT_SHA256[name]}.h5", source)
    if file_hash(source) != f"sha256:{INPUT_SHA256[name]}":
        raise RuntimeError(f"{name}: downloaded file does not match the pinned sha256")

    fixed = workdir / f"{name}.h5"
    shutil.copyfile(source, fixed)
    changes = new_attributes(name)
    with h5py.File(fixed, "r+") as f:
        for key, value in changes.items():
            f.attrs[key] = value

    # The data must be identical: this script only corrects metadata.
    if not same_arrays(read_arrays(source), read_arrays(fixed)):
        raise RuntimeError(f"{name}: arrays changed")
    with h5py.File(fixed, "r") as f:
        for key, value in changes.items():
            stored = f.attrs[key]
            stored = stored.decode() if isinstance(stored, bytes) else str(stored)
            if stored != value:
                raise RuntimeError(f"{name}: attribute {key} not written")
    return fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the fixed files and update datasets.json")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = {e["name"]: e for e in manifest["datasets"]}

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name in sorted(INPUT_SHA256):
            entry = entries[name]
            if entry["sha256"] != INPUT_SHA256[name]:
                # The manifest already points to another version, for example
                # the output of an earlier run of this script.
                print(f"skip  {name}: manifest sha256 is not the 0.1.4 input")
                continue
            fixed = fix_one(name, entry, manifest["cdn_url"], workdir)
            new_sha = file_hash(fixed).split(":", 1)[1]
            print(f"fixed {name}: {INPUT_SHA256[name][:12]} -> {new_sha[:12]}")
            if args.write:
                DATA_SIA_DIR.mkdir(exist_ok=True)
                DATA_DIR.mkdir(exist_ok=True)
                shutil.copyfile(fixed, DATA_SIA_DIR / f"{name}.h5")
                shutil.copyfile(fixed, DATA_DIR / f"{new_sha}.h5")
                entry["sha256"] = new_sha
                entry["filename"] = f"{new_sha}.h5"
                entry["etag"] = None

    if args.write:
        # Same formatting as the existing manifest (2-space indent, ASCII, final newline).
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        print(f"updated {MANIFEST_PATH}")
    else:
        print("dry run: pass --write to save the files and update the manifest")


if __name__ == "__main__":
    main()
