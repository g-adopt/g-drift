#!/usr/bin/env python3
"""Generate all example thumbnails for the documentation site.

Run this before building the docs to ensure thumbnail images are up to date.
Each example directory contains a generate_thumbnail.py script that produces
a PNG in docs/assets/images/thumbnails/.

Usage:
    python scripts/generate_thumbnails.py
"""
import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Example directories that have thumbnail generators
THUMBNAIL_DEMOS = [
    "mantle_solidus",
    "anelasticity_corrections",
    "geodynamic_adiabat",
    "linearisation",
    "tomography_models",
    "temperature_to_vs",
]


def run_thumbnail_script(demo_name):
    """Run a single thumbnail generation script."""
    script = EXAMPLES_DIR / demo_name / "generate_thumbnail.py"
    if not script.exists():
        print(f"  SKIP {demo_name}: no generate_thumbnail.py")
        return False

    print(f"  Generating {demo_name}...")
    try:
        spec = importlib.util.spec_from_file_location(
            f"thumbnail_{demo_name}", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate()
        return True
    except Exception as e:
        print(f"  FAIL {demo_name}: {e}")
        return False


def main():
    print("Generating example thumbnails")
    print("=" * 40)

    success, fail = 0, 0
    for demo in THUMBNAIL_DEMOS:
        if run_thumbnail_script(demo):
            success += 1
        else:
            fail += 1

    print("=" * 40)
    print(f"Done: {success} generated, {fail} failed")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
