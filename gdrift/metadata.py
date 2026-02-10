"""Metadata extraction and validation utilities for dataset enrichment.

This module provides functions for:
1. Extracting metadata from HDF5 files (file-level attributes)
2. Validating DOI format
3. Parsing DOIs from text (web search results)
4. Formatting citations
5. Checking metadata completeness
6. Getting SLB database references

Used primarily by the metadata enrichment pipeline (scripts/enrich_datasets_metadata.py)
to populate datasets.json with complete citation information.
"""

import re
import h5py
from typing import Dict, Optional, Any, List
from pathlib import Path


def extract_hdf5_file_attrs(file_path: str) -> Dict[str, Any]:
    """Extract file-level attributes from an HDF5 file.

    Reads the top-level attributes stored in an HDF5 file. These typically
    include metadata like author, DOI, composition, chemical system, etc.
    for thermodynamic models.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file to read.

    Returns
    -------
    dict
        Dictionary of attribute names to values. Returns empty dict if file
        doesn't exist or has no attributes.

    Examples
    --------
    >>> attrs = extract_hdf5_file_attrs("gdrift/data/model.h5")
    >>> print(attrs.get("author"))
    >>> print(attrs.get("doi"))

    Notes
    -----
    Only reads file-level attributes (f.attrs), not dataset attributes.
    All attribute values are returned as-is (typically strings or numbers).
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    try:
        with h5py.File(file_path, 'r') as f:
            # Extract all file-level attributes
            attrs = dict(f.attrs)
            # Convert bytes to strings if needed
            for key, value in attrs.items():
                if isinstance(value, bytes):
                    attrs[key] = value.decode('utf-8')
            return attrs
    except (OSError, IOError) as e:
        print(f"Warning: Could not read HDF5 file {file_path}: {e}")
        return {}


def validate_doi(doi: str) -> bool:
    """Check if a string is a valid DOI format.

    Validates DOI format according to the standard pattern:
    10.xxxx/xxxxx where xxxx is 4+ digits and xxxxx is any non-whitespace.

    Parameters
    ----------
    doi : str
        DOI string to validate (may include "https://doi.org/" prefix).

    Returns
    -------
    bool
        True if valid DOI format, False otherwise.

    Examples
    --------
    >>> validate_doi("10.1093/gji/ggaa605")
    True
    >>> validate_doi("https://doi.org/10.1016/j.epsl.2008.08.012")
    True
    >>> validate_doi("not-a-doi")
    False
    """
    if not doi:
        return False
    # Remove common prefixes
    doi_clean = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    # DOI pattern: 10.xxxx/xxxxx
    pattern = r'^10\.\d{4,}/[^\s]+$'
    return bool(re.match(pattern, doi_clean))


def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract the first DOI found in a text string.

    Searches for DOI patterns in text (e.g., from web search results).
    Returns the first match found.

    Parameters
    ----------
    text : str
        Text to search for DOI patterns.

    Returns
    -------
    str or None
        First DOI found, or None if no DOI found.

    Examples
    --------
    >>> extract_doi_from_text("See paper at https://doi.org/10.1093/gji/ggaa605")
    '10.1093/gji/ggaa605'
    >>> extract_doi_from_text("DOI: 10.1016/j.epsl.2008.08.012 for details")
    '10.1016/j.epsl.2008.08.012'
    """
    if not text:
        return None

    # Pattern to match DOI: 10.xxxx/xxxxx
    pattern = r'10\.\d{4,}/[^\s,;\)\]"]+'
    matches = re.findall(pattern, text)

    if matches:
        # Return first match, clean up any trailing punctuation
        doi = matches[0].rstrip('.,;:')
        return doi
    return None


def format_citation(dataset_dict: Dict[str, Any]) -> str:
    """Format a dataset dictionary into a citation string.

    Creates a formatted citation from dataset metadata fields.
    Format: "Author (Year). Source. DOI: https://doi.org/xxx"

    Parameters
    ----------
    dataset_dict : dict
        Dataset dictionary with fields: author, year, source, doi.

    Returns
    -------
    str
        Formatted citation string.

    Examples
    --------
    >>> ds = {
    ...     "author": "Stixrude, L., & Lithgow-Bertelloni, C.",
    ...     "year": 2021,
    ...     "source": "Thermal expansivity...",
    ...     "doi": "10.1093/gji/ggaa605"
    ... }
    >>> print(format_citation(ds))
    Stixrude, L., & Lithgow-Bertelloni, C. (2021). Thermal expansivity... DOI: https://doi.org/10.1093/gji/ggaa605
    """
    parts = []

    # Author (Year)
    if dataset_dict.get("author"):
        author_year = dataset_dict["author"]
        if dataset_dict.get("year"):
            author_year += f" ({dataset_dict['year']})"
        parts.append(author_year)

    # Source
    if dataset_dict.get("source"):
        parts.append(dataset_dict["source"])

    # DOI
    if dataset_dict.get("doi"):
        doi = dataset_dict["doi"]
        if validate_doi(doi):
            doi_clean = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            parts.append(f"DOI: https://doi.org/{doi_clean}")

    return ". ".join(parts) if parts else "Citation information incomplete"


def check_metadata_completeness(dataset_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if a dataset has complete metadata for catalog inclusion.

    A dataset is considered complete if it has:
    - DOI or complete author/year information
    - Source/citation string
    - Description

    Parameters
    ----------
    dataset_dict : dict
        Dataset dictionary to check.

    Returns
    -------
    tuple of (bool, list of str)
        - is_complete: True if all required fields present
        - missing_fields: List of missing field names

    Examples
    --------
    >>> ds = {"name": "SLB_24_pyrolite", "doi": "10.xxx/yyy", "source": "..."}
    >>> complete, missing = check_metadata_completeness(ds)
    >>> print(f"Complete: {complete}, Missing: {missing}")
    """
    required_fields = {
        "doi": "DOI",
        "source": "Citation/Source",
        "description": "Description"
    }

    # Alternative to DOI: complete author + year
    has_doi = dataset_dict.get("doi") and validate_doi(dataset_dict.get("doi"))
    has_author_year = dataset_dict.get("author") and dataset_dict.get("year")

    missing = []

    # Check DOI or author+year
    if not has_doi and not has_author_year:
        missing.append("DOI or Author+Year")

    # Check other required fields
    for field, label in required_fields.items():
        if field == "doi":
            continue  # Already checked above
        if not dataset_dict.get(field):
            missing.append(label)

    is_complete = len(missing) == 0
    return is_complete, missing


def get_slb_reference(version: str) -> Dict[str, str]:
    """Get bibliographic reference for SLB database versions.

    Returns standard citation information for Stixrude & Lithgow-Bertelloni
    thermodynamic database versions.

    Parameters
    ----------
    version : str
        SLB version: "08", "11", "16", "21", or "24".

    Returns
    -------
    dict
        Dictionary with keys: doi, year, author, citation.

    Examples
    --------
    >>> ref = get_slb_reference("21")
    >>> print(ref["doi"])
    10.1093/gji/ggaa605

    Raises
    ------
    ValueError
        If version is not recognized.
    """
    references = {
        "08": {
            "doi": "10.1016/j.epsl.2008.08.012",
            "year": 2008,
            "author": "Xu, W., Lithgow-Bertelloni, C., Stixrude, L., & Ritsema, J.",
            "citation": "Xu, W.; Lithgow-Bertelloni, C.; Stixrude, L.; Ritsema, J. \"The effect of bulk composition and temperature on mantle seismic structure\", Earth and Planetary Science Letters, 2008, 275, 70-79."
        },
        "11": {
            "doi": "10.1111/j.1365-246X.2010.04890.x",
            "year": 2011,
            "author": "Stixrude, L., & Lithgow-Bertelloni, C.",
            "citation": "Stixrude, L.; Lithgow-Bertelloni, C. \"Thermodynamics of mantle minerals -- II. Phase equilibria\", Geophysical Journal International, 2011, 184, 1180-1213."
        },
        "16": {
            "doi": "10.1093/gji/ggw100",
            "year": 2016,
            "author": "Stixrude, L., & Lithgow-Bertelloni, C.",
            "citation": "Stixrude, L., & Lithgow-Bertelloni, C. (2016). Thermodynamics of mantle minerals-I. Physical properties. Geophysical Journal International, 206(2), 1176-1199."
        },
        "21": {
            "doi": "10.1093/gji/ggaa605",
            "year": 2021,
            "author": "Stixrude, L., & Lithgow-Bertelloni, C.",
            "citation": "Stixrude, L.; Lithgow-Bertelloni, C. \"Thermal expansivity, heat capacity and bulk modulus of the mantle\", Geophysical Journal International, 2021, 228, 1119-1149."
        },
        "24": {
            "doi": "10.1093/gji/ggae178",
            "year": 2024,
            "author": "Stixrude, L., & Lithgow-Bertelloni, C.",
            "citation": "Stixrude, L.; Lithgow-Bertelloni, C. \"Thermodynamics of mantle minerals -- III. The role of iron\", Geophysical Journal International, 2024, 238, 1056-1079."
        }
    }

    if version not in references:
        raise ValueError(f"Unknown SLB version: {version}. Valid versions: {list(references.keys())}")

    return references[version]


def parse_composition_from_name(dataset_name: str) -> Optional[Dict[str, str]]:
    """Parse composition and chemical system from dataset name.

    Extracts composition (pyrolite/depleted-mantle/bulk-oceanic-crust)
    and chemical system (MS/FMS/FMAS/CFMS/CFMAS/NCFMAS/NCMAS) from
    dataset name patterns like "SLB_21_pyroliteCFMAS".

    Parameters
    ----------
    dataset_name : str
        Dataset name to parse.

    Returns
    -------
    dict or None
        Dictionary with 'composition' and 'chemical_system' keys, or None
        if pattern not recognized.

    Examples
    --------
    >>> parse_composition_from_name("SLB_21_pyroliteCFMAS")
    {'composition': 'pyrolite', 'chemical_system': 'CFMAS'}
    >>> parse_composition_from_name("SLB_08_depleted-mantleMS")
    {'composition': 'depleted-mantle', 'chemical_system': 'MS'}
    """
    # Pattern: SLB_XX_<composition><chemical_system>
    pattern = r'SLB_\d+_(pyrolite|depleted-mantle|bulk-oceanic-crust)(MS|FMS|FMAS|CFMS|CFMAS|NCFMAS|NCMAS)$'
    match = re.search(pattern, dataset_name)

    if match:
        return {
            "composition": match.group(1),
            "chemical_system": match.group(2)
        }
    return None


def extract_year_from_citation(citation: str) -> Optional[int]:
    """Extract publication year from a citation string.

    Searches for 4-digit years (1900-2030) in citation text.

    Parameters
    ----------
    citation : str
        Citation string to parse.

    Returns
    -------
    int or None
        First valid year found, or None.

    Examples
    --------
    >>> extract_year_from_citation("Smith et al. (2021). Title...")
    2021
    >>> extract_year_from_citation("Science 2019, 364, 1234-1239")
    2019
    """
    if not citation:
        return None

    # Find 4-digit years in reasonable range
    pattern = r'\b(19\d{2}|20[0-2]\d|2030)\b'
    matches = re.findall(pattern, citation)

    if matches:
        return int(matches[0])
    return None
