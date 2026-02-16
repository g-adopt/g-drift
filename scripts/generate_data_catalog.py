#!/usr/bin/env python3
"""Generate data catalog markdown files from datasets.json manifest.

This script reads the enriched datasets.json and generates two markdown files:
1. data-catalog-generated.md: Complete metadata (for inclusion in docs)
2. data-catalog-incomplete.md: Datasets needing metadata (for engineers)

The script is designed to run in CI and complete in <5 seconds. It requires
no HDF5 file access - only reads the JSON manifest.

Usage:
    python scripts/generate_data_catalog.py

Outputs:
    docs/data-catalog-generated.md
    docs/data-catalog-incomplete.md

Architecture:
    - Fast, deterministic, offline-friendly
    - Reads enriched JSON only
    - Groups datasets by type
    - Formats citations with DOI links
    - Generates summary statistics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gdrift.metadata import format_citation, validate_doi


class DatasetCatalogGenerator:
    """Generate markdown documentation from datasets.json."""

    def __init__(self, manifest_path: Path):
        """Initialize generator.

        Parameters
        ----------
        manifest_path : Path
            Path to datasets.json file.
        """
        self.manifest_path = manifest_path

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.datasets = self.manifest["datasets"]

    def group_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group datasets by type.

        Returns
        -------
        dict
            Dictionary mapping type names to lists of dataset entries.
        """
        grouped = defaultdict(list)

        for dataset in self.datasets:
            dtype = dataset.get("type", "UNKNOWN")
            grouped[dtype].append(dataset)

        return dict(grouped)

    def format_citation_with_link(self, dataset: Dict[str, Any]) -> str:
        """Format citation with DOI as clickable link.

        Parameters
        ----------
        dataset : dict
            Dataset entry.

        Returns
        -------
        str
            Formatted citation string with markdown DOI link.
        """
        parts = []

        # Author (Year)
        if dataset.get("author"):
            author_year = dataset["author"]
            if dataset.get("year"):
                author_year += f" ({dataset['year']})"
            parts.append(author_year)

        # Source/Title
        if dataset.get("source"):
            # Truncate long sources
            source = dataset["source"]
            if len(source) > 150:
                source = source[:147] + "..."
            parts.append(source)

        # DOI as link
        if dataset.get("doi"):
            doi = dataset["doi"]
            if validate_doi(doi):
                doi_clean = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                parts.append(f"[DOI: {doi_clean}](https://doi.org/{doi_clean})")

        return ". ".join(parts) if parts else "Citation information incomplete"

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the catalog.

        Returns
        -------
        dict
            Summary statistics including counts by type, completeness, etc.
        """
        stats = {
            "total": len(self.datasets),
            "by_type": {},
            "complete": 0,
            "incomplete": 0,
            "with_doi": 0,
            "without_doi": 0,
            "by_source": {}
        }

        for dataset in self.datasets:
            # Type counts
            dtype = dataset.get("type", "UNKNOWN")
            stats["by_type"][dtype] = stats["by_type"].get(dtype, 0) + 1

            # Completeness
            if dataset.get("metadata_complete"):
                stats["complete"] += 1
            else:
                stats["incomplete"] += 1

            # DOI coverage
            if dataset.get("doi") and validate_doi(dataset["doi"]):
                stats["with_doi"] += 1
            else:
                stats["without_doi"] += 1

            # Metadata source
            source = dataset.get("metadata_source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        return stats

    def generate_complete_catalog(self, output_path: Path):
        """Generate markdown for datasets with complete metadata.

        Parameters
        ----------
        output_path : Path
            Output file path for data-catalog-generated.md.
        """
        lines = []

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        lines.append("# Data Catalog")
        lines.append("")
        lines.append(f"*Auto-generated on {timestamp}*")
        lines.append("")

        # Summary statistics
        stats = self.get_summary_statistics()
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Total datasets:** {stats['total']}")
        lines.append(f"**Complete metadata:** {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
        lines.append(f"**With DOI:** {stats['with_doi']} ({stats['with_doi']/stats['total']*100:.1f}%)")
        lines.append("")

        # Summary table by type
        lines.append("### Datasets by Type")
        lines.append("")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")

        for dtype, count in sorted(stats["by_type"].items()):
            # Convert enum name to readable
            readable = dtype.replace("_", " ").title()
            lines.append(f"| {readable} | {count} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Group datasets and generate detailed listings
        grouped = self.group_by_type()

        # Define order for display
        type_order = [
            "EARTH_MODEL",
            "SOLIDUS_PROFILE",
            "GEODYNAMIC_PROFILE",
            "THERMODYNAMIC_MODEL",
            "TOMOGRAPHY_MODEL"
        ]

        type_names = {
            "EARTH_MODEL": "Reference Earth Models",
            "SOLIDUS_PROFILE": "Solidus Temperature Profiles",
            "GEODYNAMIC_PROFILE": "Geodynamic Profiles",
            "THERMODYNAMIC_MODEL": "Thermodynamic Models",
            "TOMOGRAPHY_MODEL": "Seismic Tomography Models"
        }

        for dtype in type_order:
            if dtype not in grouped:
                continue

            datasets = grouped[dtype]
            count = len(datasets)

            lines.append(f"## {type_names[dtype]} ({count} datasets)")
            lines.append("")

            # Sort by name
            datasets_sorted = sorted(datasets, key=lambda d: d["name"])

            for dataset in datasets_sorted:
                # Only include datasets with complete metadata
                if not dataset.get("metadata_complete"):
                    continue

                lines.append(f"### {dataset['name']}")
                lines.append("")

                # Citation
                citation = self.format_citation_with_link(dataset)
                lines.append(f"**Citation:** {citation}")
                lines.append("")

                # Description
                if dataset.get("description"):
                    lines.append(f"**Description:** {dataset['description']}")
                    lines.append("")

                # Composition (for thermodynamic models)
                if dataset.get("composition"):
                    lines.append(f"**Composition:** {dataset['composition']}")
                    if dataset.get("chemical_system"):
                        lines.append(f"**Chemical System:** {dataset['chemical_system']}")
                    lines.append("")

                # Usage
                if dataset.get("utility"):
                    utility = dataset["utility"]
                    lines.append(f"**Usage:** `gdrift.{utility}(\"{dataset['name']}\")`")
                    lines.append("")

                # File info
                filename = dataset.get("filename", "N/A")
                lines.append(f"**File:** `{filename}`")
                lines.append("")

                lines.append("---")
                lines.append("")

        # Footer
        lines.append("## Notes")
        lines.append("")
        lines.append("- This catalog is auto-generated from `gdrift/datasets.json`")
        lines.append("- Citation information is extracted from HDF5 files and web sources")
        lines.append("- All datasets listed have complete metadata with proper citations")
        lines.append("")

        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✅ Generated: {output_path}")

    def generate_incomplete_catalog(self, output_path: Path):
        """Generate markdown for datasets with incomplete metadata.

        Parameters
        ----------
        output_path : Path
            Output file path for data-catalog-incomplete.md.
        """
        lines = []

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        lines.append("# Datasets Needing Metadata")
        lines.append("")
        lines.append(f"*Auto-generated on {timestamp}*")
        lines.append("")
        lines.append("The following datasets have incomplete metadata and need attention.")
        lines.append("")

        # Filter incomplete
        incomplete = [d for d in self.datasets if not d.get("metadata_complete")]

        if not incomplete:
            lines.append("✅ **All datasets have complete metadata!**")
        else:
            lines.append(f"**Total incomplete:** {len(incomplete)} / {len(self.datasets)}")
            lines.append("")

            # Group by what's missing
            lines.append("## Action Items")
            lines.append("")

            for dataset in sorted(incomplete, key=lambda d: d["name"]):
                lines.append(f"### {dataset['name']}")
                lines.append("")

                # What's present
                present = []
                if dataset.get("doi"):
                    present.append("DOI")
                if dataset.get("author"):
                    present.append("Author")
                if dataset.get("year"):
                    present.append("Year")
                if dataset.get("description"):
                    present.append("Description")

                if present:
                    lines.append(f"**Present:** {', '.join(present)}")
                    lines.append("")

                # What's missing
                missing = []
                if not dataset.get("doi"):
                    missing.append("❌ DOI")
                if not dataset.get("author"):
                    missing.append("❌ Author")
                if not dataset.get("year"):
                    missing.append("❌ Year")
                if not dataset.get("description"):
                    missing.append("❌ Description")

                if missing:
                    lines.append(f"**Missing:** {', '.join(missing)}")
                    lines.append("")

                # Current source/citation
                if dataset.get("source"):
                    lines.append(f"**Current citation:** {dataset['source']}")
                    lines.append("")

                # Search suggestions
                if "3d_seismic_" in dataset["name"]:
                    model_name = dataset["name"].replace("3d_seismic_", "")
                    lines.append(f"**Search suggestion:** \"{model_name} seismic tomography DOI\"")
                    lines.append("")

                lines.append("---")
                lines.append("")

        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✅ Generated: {output_path}")


def main():
    """Main entry point."""
    # Paths
    manifest_path = Path(__file__).parent.parent / "gdrift" / "datasets.json"
    docs_dir = Path(__file__).parent.parent / "docs"

    # Create docs directory if needed
    docs_dir.mkdir(exist_ok=True)

    output_complete = docs_dir / "data-catalog-generated.md"
    output_incomplete = docs_dir / "data-catalog-incomplete.md"

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    print("="*80)
    print("DATA CATALOG GENERATOR")
    print("="*80)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {docs_dir}")
    print("="*80)

    # Generate catalogs
    generator = DatasetCatalogGenerator(manifest_path)

    print("\n📝 Generating complete catalog...")
    generator.generate_complete_catalog(output_complete)

    print("\n📝 Generating incomplete catalog...")
    generator.generate_incomplete_catalog(output_incomplete)

    # Summary
    stats = generator.get_summary_statistics()

    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"Total datasets: {stats['total']}")
    print(f"Complete metadata: {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
    print(f"Incomplete metadata: {stats['incomplete']}")
    print(f"DOI coverage: {stats['with_doi']} / {stats['total']} ({stats['with_doi']/stats['total']*100:.1f}%)")
    print("")
    print("Metadata sources:")
    for source, count in sorted(stats["by_source"].items()):
        print(f"  - {source}: {count}")
    print("="*80)

    print("\n✅ Catalog generation complete!")
    print(f"   Complete: {output_complete}")
    print(f"   Incomplete: {output_incomplete}")


if __name__ == "__main__":
    main()
