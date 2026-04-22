"""Regenerate expected.pkl for the coastlines_vtp demo.

Run this after intentional changes to the triangulation pipeline:

    cd examples/coastlines_vtp
    python generate_expected.py
"""
import pickle
from pathlib import Path

import pyvista as pv

DEMO_DIR = Path(__file__).parent

namespace = {"__file__": str(DEMO_DIR / "demo.py")}
exec(open(DEMO_DIR / "demo.py").read(), namespace)

writer = namespace["writer"]
ages = namespace["ages_ma"]

mesh_age_0 = pv.read(writer.vtp_dir / "coastlines_0.vtp")

expected = {
    "ages": list(ages),
    "n_ages": len(ages),
    "n_cells_age_0": int(mesh_age_0.n_cells),
}

with open(DEMO_DIR / "expected.pkl", "wb") as f:
    pickle.dump(expected, f)

print(f"Wrote {DEMO_DIR / 'expected.pkl'}")
for k, v in expected.items():
    print(f"  {k}: {v}")
