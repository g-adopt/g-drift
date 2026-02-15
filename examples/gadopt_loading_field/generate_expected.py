#!/usr/bin/env python
"""Generate expected.pkl for gadopt loading field demo regression tests.

Requires gadopt (Firedrake). Run with the Firedrake-enabled Python:

    /Users/sghelichkhani/Workplace/firedrake-2026-01-13/venv-firedrake/bin/python3 generate_expected.py

Or in CI, within the Firedrake container after activating the venv.

Usage:
    python generate_expected.py
"""

import pickle
from pathlib import Path

import numpy as np


def main():
    demo_dir = Path(__file__).parent

    # Run the demo to capture computed values
    namespace = {"__file__": str(demo_dir / "demo.py")}
    exec(open(demo_dir / "demo.py").read(), namespace)

    vs_data = namespace["vs"].dat.data_with_halos
    t_data = namespace["temperature"].dat.data_with_halos

    expected = {
        "n_vertices": len(vs_data),
        "vs_min": float(vs_data.min()),
        "vs_max": float(vs_data.max()),
        "vs_mean": float(vs_data.mean()),
        "temperature_min": float(t_data.min()),
        "temperature_max": float(t_data.max()),
        "temperature_mean": float(t_data.mean()),
    }

    output_file = demo_dir / "expected.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expected, f)

    print(f"Generated {output_file}")
    for key, val in expected.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
