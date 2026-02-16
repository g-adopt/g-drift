#!/usr/bin/env python3
"""
Batch convert all MMA-EoS output files to HDF5 format.

This script discovers all SLB datasets in the output-SLB* directories,
converts them to HDF5, and generates a manifest entries JSON file
for manual merging into datasets.json.
"""

import hashlib
import json
from pathlib import Path

from convert_slb_to_hdf5 import convert_one_model, SLB_REFERENCES


def file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def discover_datasets(base_dir):
    """Scan output-SLB* directories for .out files."""
    datasets = []
    for slb_dir in sorted(base_dir.glob('output-SLB*')):
        version = slb_dir.name.replace('output-SLB', '')
        for comp_dir in sorted(slb_dir.iterdir()):
            if not comp_dir.is_dir():
                continue
            composition = comp_dir.name
            for sys_dir in sorted(comp_dir.iterdir()):
                if not sys_dir.is_dir():
                    continue
                system = sys_dir.name
                if (sys_dir / 'prop.out').exists() and (sys_dir / 'opti.out').exists():
                    datasets.append({
                        'version': version,
                        'composition': composition,
                        'system': system,
                        'source_dir': sys_dir
                    })
    return datasets


def main():
    base_dir = Path('gdrift/data-sia')
    output_dir = base_dir

    datasets = discover_datasets(base_dir)
    print(f"Found {len(datasets)} datasets to convert\n")

    manifest_entries = []
    success_count = 0
    failed_datasets = []

    for i, ds in enumerate(datasets, 1):
        version = ds['version']
        composition = ds['composition']
        system = ds['system']
        source_dir = ds['source_dir']

        try:
            print(f"[{i}/{len(datasets)}] ", end='')

            # Convert
            output_file = convert_one_model(
                source_dir,
                output_dir,
                version,
                composition,
                system
            )

            # Compute hash
            hash_val = file_hash(output_file)

            # Build manifest entry
            ref = SLB_REFERENCES[version]
            manifest_entries.append({
                'name': f"SLB_{version}_{composition}{system.upper()}",
                'filename': output_file.name,
                'type': 'THERMODYNAMIC_MODEL',
                'utility': 'THERMODYNAMIC',
                'source': ref['citation'],
                'doi': ref['doi'] if ref['doi'] else '',
                'year': ref['year'],
                'description': f"SLB {version} thermodynamic model for {composition} ({system.upper()})",
                'sha256': hash_val,
            })

            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_datasets.append({
                'version': version,
                'composition': composition,
                'system': system,
                'error': str(e)
            })

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {success_count}/{len(datasets)}")
    print(f"  Failed: {len(failed_datasets)}/{len(datasets)}")

    if failed_datasets:
        print(f"\nFailed datasets:")
        for fd in failed_datasets:
            print(f"  - SLB_{fd['version']}_{fd['composition']}{fd['system'].upper()}: {fd['error']}")

    # Save manifest entries for manual merge
    manifest_file = 'new_manifest_entries.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"\nManifest entries saved to: {manifest_file}")
    print(f"Manually merge these into gdrift/datasets.json")


if __name__ == '__main__':
    main()
