"""Print the sha256 of every HDF5 file in the package directory gdrift/data/.

This is a helper for maintainers, not a test. Run it as a script:

    python tests/test_hash.py

The code runs only as a script, because pytest imports every tests/test_*.py
file, and hashing all local datasets (several GB) at each test run is slow.
"""
from pathlib import Path

from gdrift.io import file_hash, DATA_PATH


if __name__ == "__main__":
    data_path = Path(DATA_PATH).resolve()

    for i in data_path.glob("*.h5"):
        print(f'"{i.name}": "{file_hash(i)}",')
