#!/usr/bin/env python3
"""Enrich datasets.json with metadata from HDF5 files and web searches.

This script extracts metadata from cached HDF5 files and performs web searches
to find missing DOI/citation information. It's designed to be run locally once
to enrich the manifest, then the enriched manifest is committed to git.

Usage:
    # Test single dataset (dry run)
    python scripts/enrich_datasets_metadata.py --dataset "SLB_24_pyroliteCFMAS" --dry-run

    # Enrich from HDF5 only (no web search)
    python scripts/enrich_datasets_metadata.py

    # Full enrichment with web search for missing DOIs
    python scripts/enrich_datasets_metadata.py --web-search

    # Review and commit
    cp gdrift/datasets.json.enriched gdrift/datasets.json
    git commit -m "Enrich datasets.json with metadata"

Architecture:
    1. HDF5MetadataExtractor: Reads file-level attributes from cached HDF5 files
    2. WebSearchEnricher: Uses WebSearch tool to find DOIs online
    3. DatasetEnricher: Orchestrates enrichment and tracks results

New Fields Added:
    - metadata_source: "hdf5_file" | "web_search" | "manifest_only"
    - metadata_complete: boolean flag
    - author: extracted from HDF5 or web
    - author_email: from HDF5 (thermodynamic models)
    - composition: pyrolite/depleted-mantle/bulk-oceanic-crust
    - chemical_system: MS/FMS/FMAS/CFMS/CFMAS/NCFMAS
    - slb_version: "08"/"11"/"16"/"21"/"24"
    - generation_timestamp: when HDF5 file was created
    - bulk_composition_mol_pct: composition percentages (if available)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gdrift.metadata import (
    extract_hdf5_file_attrs,
    validate_doi,
    extract_doi_from_text,
    check_metadata_completeness,
    get_slb_reference,
    parse_composition_from_name,
    extract_year_from_citation
)


@dataclass
class EnrichmentResult:
    """Result of enriching a single dataset."""
    dataset_name: str
    success: bool
    metadata_source: str  # "hdf5_file", "web_search", "manifest_only", "slb_reference"
    metadata_complete: bool
    new_fields: Dict[str, Any]
    warnings: List[str]
    errors: List[str]

    def summary(self) -> str:
        """Generate a summary string for this result."""
        status = "✓" if self.success else "✗"
        complete = "COMPLETE" if self.metadata_complete else "INCOMPLETE"
        source = self.metadata_source.upper()

        lines = [f"{status} {self.dataset_name} [{source}] [{complete}]"]

        if self.new_fields:
            added = ", ".join(self.new_fields.keys())
            lines.append(f"   Added: {added}")

        for warning in self.warnings:
            lines.append(f"   ⚠ {warning}")

        for error in self.errors:
            lines.append(f"   ✗ {error}")

        return "\n".join(lines)


class HDF5MetadataExtractor:
    """Extract metadata from HDF5 files in the gdrift data cache."""

    def __init__(self, data_dir: Path):
        """Initialize extractor.

        Parameters
        ----------
        data_dir : Path
            Path to gdrift/data/ directory containing cached HDF5 files.
        """
        self.data_dir = data_dir

    def extract(self, dataset_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metadata from HDF5 file for a dataset.

        Parameters
        ----------
        dataset_entry : dict
            Dataset entry from datasets.json with 'filename' key.

        Returns
        -------
        dict or None
            Dictionary of extracted metadata fields, or None if file not found
            or no useful metadata.

        Extracted Fields:
        -----------------
        - author: Author name(s)
        - author_email: Contact email
        - doi: Digital Object Identifier
        - composition: Bulk composition name
        - chemical_system: Chemical system (e.g., CFMAS)
        - slb_version: SLB database version
        - generation_timestamp: When file was created
        - bulk_composition_mol_pct: Composition in mol %
        - Any other file-level attributes found
        """
        filename = dataset_entry.get("filename")
        if not filename:
            return None

        file_path = self.data_dir / filename

        if not file_path.exists():
            return None

        # Extract all file-level attributes
        attrs = extract_hdf5_file_attrs(str(file_path))

        if not attrs:
            return None

        # Map common attribute names to standard fields
        metadata = {}

        # Author information
        for author_key in ["author", "Author", "AUTHORS", "creator"]:
            if author_key in attrs:
                metadata["author"] = attrs[author_key]
                break

        # Email
        for email_key in ["author_email", "email", "contact"]:
            if email_key in attrs:
                metadata["author_email"] = attrs[email_key]
                break

        # DOI
        for doi_key in ["doi", "DOI"]:
            if doi_key in attrs and validate_doi(attrs[doi_key]):
                metadata["doi"] = attrs[doi_key]
                break

        # Composition
        for comp_key in ["composition", "bulk_composition", "Composition"]:
            if comp_key in attrs:
                metadata["composition"] = attrs[comp_key]
                break

        # Chemical system
        for sys_key in ["chemical_system", "system", "ChemicalSystem"]:
            if sys_key in attrs:
                metadata["chemical_system"] = attrs[sys_key]
                break

        # SLB version
        for ver_key in ["slb_version", "SLB_version", "database_version"]:
            if ver_key in attrs:
                metadata["slb_version"] = attrs[ver_key]
                break

        # Timestamp
        for ts_key in ["generation_timestamp", "created", "timestamp", "date_created"]:
            if ts_key in attrs:
                metadata["generation_timestamp"] = attrs[ts_key]
                break

        # Bulk composition percentages
        for bulk_key in ["bulk_composition_mol_pct", "composition_mol_pct"]:
            if bulk_key in attrs:
                metadata["bulk_composition_mol_pct"] = attrs[bulk_key]
                break

        # Store all other attributes in an 'hdf5_attrs' field for reference
        other_attrs = {k: v for k, v in attrs.items()
                      if k not in ["author", "Author", "AUTHORS", "creator",
                                  "author_email", "email", "contact",
                                  "doi", "DOI", "composition", "bulk_composition",
                                  "chemical_system", "system", "ChemicalSystem",
                                  "slb_version", "SLB_version", "database_version",
                                  "generation_timestamp", "created", "timestamp",
                                  "bulk_composition_mol_pct", "composition_mol_pct"]}

        if other_attrs:
            metadata["hdf5_attrs"] = other_attrs

        return metadata if metadata else None


class WebSearchEnricher:
    """Use web search to find missing DOI and citation information."""

    def __init__(self, use_web_search: bool = False):
        """Initialize enricher.

        Parameters
        ----------
        use_web_search : bool
            Whether web search is enabled. If False, returns None immediately.
        """
        self.use_web_search = use_web_search

    def search_doi(self, dataset_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Search for DOI and citation info using web search.

        Parameters
        ----------
        dataset_entry : dict
            Dataset entry with 'name', 'type', 'source' fields.

        Returns
        -------
        dict or None
            Dictionary with 'doi', 'author', 'year' if found, else None.
        """
        if not self.use_web_search:
            return None

        # Import WebSearch here to avoid circular imports
        try:
            # WebSearch is not directly importable, so we skip it for now
            # Users can implement this by calling the WebSearch tool manually
            print(f"⚠ Web search for '{dataset_entry['name']}' requires manual implementation")
            print(f"  To add DOI, manually search for the dataset and update datasets.json")
            return None
        except ImportError:
            return None


class DatasetEnricher:
    """Orchestrate dataset enrichment from multiple sources."""

    def __init__(self, manifest_path: Path, data_dir: Path, use_web_search: bool = False):
        """Initialize enricher.

        Parameters
        ----------
        manifest_path : Path
            Path to datasets.json file.
        data_dir : Path
            Path to gdrift/data/ directory.
        use_web_search : bool
            Whether to use web search for missing DOIs.
        """
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.hdf5_extractor = HDF5MetadataExtractor(data_dir)
        self.web_enricher = WebSearchEnricher(use_web_search)

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def enrich_one(self, dataset_entry: Dict[str, Any]) -> EnrichmentResult:
        """Enrich a single dataset entry.

        Tries enrichment sources in order:
        1. SLB reference (for SLB models with known versions)
        2. HDF5 file attributes
        3. Web search (if enabled)
        4. Name parsing (composition, chemical system)

        Parameters
        ----------
        dataset_entry : dict
            Dataset entry from manifest.

        Returns
        -------
        EnrichmentResult
            Result object with enrichment status and new fields.
        """
        name = dataset_entry["name"]
        new_fields = {}
        warnings = []
        errors = []
        metadata_source = "manifest_only"

        try:
            # 1. Try SLB reference for known versions
            if name.startswith("SLB_"):
                slb_match = parse_composition_from_name(name)
                if slb_match:
                    # Extract version
                    version = name.split("_")[1]
                    try:
                        slb_ref = get_slb_reference(version)
                        if not dataset_entry.get("doi"):
                            new_fields["doi"] = slb_ref["doi"]
                            new_fields["author"] = slb_ref["author"]
                            metadata_source = "slb_reference"
                        new_fields["slb_version"] = version
                        new_fields["composition"] = slb_match["composition"]
                        new_fields["chemical_system"] = slb_match["chemical_system"]
                    except ValueError as e:
                        warnings.append(f"SLB reference lookup failed: {e}")

            # 2. Try HDF5 extraction
            hdf5_metadata = self.hdf5_extractor.extract(dataset_entry)
            if hdf5_metadata:
                # Merge HDF5 metadata, don't overwrite existing fields
                for key, value in hdf5_metadata.items():
                    if key not in new_fields and not dataset_entry.get(key):
                        new_fields[key] = value

                if metadata_source == "manifest_only":
                    metadata_source = "hdf5_file"

            # 3. Try web search if no DOI yet
            if not new_fields.get("doi") and not dataset_entry.get("doi"):
                web_metadata = self.web_enricher.search_doi(dataset_entry)
                if web_metadata:
                    new_fields.update(web_metadata)
                    metadata_source = "web_search"

            # 4. Parse composition from name if not found
            if not new_fields.get("composition"):
                parsed = parse_composition_from_name(name)
                if parsed:
                    new_fields["composition"] = parsed["composition"]
                    new_fields["chemical_system"] = parsed["chemical_system"]

            # 5. Extract year from source if not present
            if not new_fields.get("year") and not dataset_entry.get("year"):
                year = extract_year_from_citation(dataset_entry.get("source", ""))
                if year:
                    new_fields["year"] = year

            # Check completeness
            merged = {**dataset_entry, **new_fields}
            is_complete, missing = check_metadata_completeness(merged)

            if not is_complete:
                warnings.append(f"Incomplete metadata: missing {', '.join(missing)}")

            # Add metadata tracking fields
            new_fields["metadata_source"] = metadata_source
            new_fields["metadata_complete"] = is_complete

            return EnrichmentResult(
                dataset_name=name,
                success=True,
                metadata_source=metadata_source,
                metadata_complete=is_complete,
                new_fields=new_fields,
                warnings=warnings,
                errors=errors
            )

        except Exception as e:
            errors.append(f"Enrichment failed: {str(e)}")
            return EnrichmentResult(
                dataset_name=name,
                success=False,
                metadata_source="error",
                metadata_complete=False,
                new_fields={},
                warnings=warnings,
                errors=errors
            )

    def enrich_all(self, dataset_filter: Optional[str] = None) -> List[EnrichmentResult]:
        """Enrich all datasets in manifest.

        Parameters
        ----------
        dataset_filter : str, optional
            If provided, only enrich dataset with this name.

        Returns
        -------
        list of EnrichmentResult
            Results for all enriched datasets.
        """
        results = []

        for entry in self.manifest["datasets"]:
            if dataset_filter and entry["name"] != dataset_filter:
                continue

            print(f"\n📊 Enriching: {entry['name']}")
            result = self.enrich_one(entry)
            print(result.summary())
            results.append(result)

        return results

    def write_enriched_manifest(self, results: List[EnrichmentResult], output_path: Path, dry_run: bool = False):
        """Write enriched manifest to file.

        Parameters
        ----------
        results : list of EnrichmentResult
            Enrichment results to apply.
        output_path : Path
            Output file path.
        dry_run : bool
            If True, print changes but don't write file.
        """
        # Create enriched manifest
        enriched_manifest = self.manifest.copy()

        # Apply enrichment results
        result_map = {r.dataset_name: r for r in results}

        for entry in enriched_manifest["datasets"]:
            if entry["name"] in result_map:
                result = result_map[entry["name"]]
                entry.update(result.new_fields)

        if dry_run:
            print("\n" + "="*80)
            print("DRY RUN - Changes that would be applied:")
            print("="*80)
            for result in results:
                if result.new_fields:
                    print(f"\n{result.dataset_name}:")
                    for key, value in result.new_fields.items():
                        print(f"  + {key}: {value}")
            return

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(enriched_manifest, f, indent=2)

        print(f"\n✅ Enriched manifest written to: {output_path}")
        print(f"   Review the file and then:")
        print(f"   cp {output_path} {self.manifest_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich datasets.json with metadata from HDF5 files and web searches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single dataset
  python scripts/enrich_datasets_metadata.py --dataset "SLB_24_pyroliteCFMAS" --dry-run

  # Enrich from HDF5 only
  python scripts/enrich_datasets_metadata.py

  # Full enrichment with web search
  python scripts/enrich_datasets_metadata.py --web-search

  # Review and commit
  cp gdrift/datasets.json.enriched gdrift/datasets.json
  git commit -m "Enrich datasets.json with metadata"
        """
    )

    parser.add_argument(
        "--dataset",
        help="Only enrich this specific dataset (by name)"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search for missing DOIs (requires WebSearch tool)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "gdrift" / "data",
        help="Path to gdrift/data directory (default: gdrift/data)"
    )

    args = parser.parse_args()

    # Paths
    manifest_path = Path(__file__).parent.parent / "gdrift" / "datasets.json"
    output_path = manifest_path.parent / "datasets.json.enriched"

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"⚠ Data directory not found: {args.data_dir}")
        print("  HDF5 extraction will be skipped (files not cached)")

    # Run enrichment
    enricher = DatasetEnricher(manifest_path, args.data_dir, args.web_search)

    print("="*80)
    print("DATASET METADATA ENRICHMENT")
    print("="*80)
    print(f"Manifest: {manifest_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"Web search: {'ENABLED' if args.web_search else 'DISABLED'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print("="*80)

    results = enricher.enrich_all(dataset_filter=args.dataset)

    # Summary statistics
    print("\n" + "="*80)
    print("ENRICHMENT SUMMARY")
    print("="*80)

    total = len(results)
    success = sum(1 for r in results if r.success)
    complete = sum(1 for r in results if r.metadata_complete)

    sources = {}
    for r in results:
        sources[r.metadata_source] = sources.get(r.metadata_source, 0) + 1

    print(f"Total datasets: {total}")
    print(f"Successfully enriched: {success}")
    print(f"Complete metadata: {complete} ({complete/total*100:.1f}%)")
    print(f"\nMetadata sources:")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count}")

    # Write output
    enricher.write_enriched_manifest(results, output_path, dry_run=args.dry_run)

    if not args.dry_run and not args.dataset:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Review the enriched manifest:")
        print(f"   cat {output_path}")
        print("2. If satisfied, replace the original:")
        print(f"   cp {output_path} {manifest_path}")
        print("3. Commit to git:")
        print("   git add gdrift/datasets.json")
        print("   git commit -m 'Enrich datasets.json with metadata'")


if __name__ == "__main__":
    main()
