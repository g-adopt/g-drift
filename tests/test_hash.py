from gdrift.io import file_hash, DATA_PATH
from pathlib import Path

data_path = Path(DATA_PATH).resolve()

for i in data_path.glob("*.h5"):
    print(f'"{i.name}": "{file_hash(i)}",')
