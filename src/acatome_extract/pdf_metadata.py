"""PDF metadata enrichment: write DOI, title, authors back to PDF files via exiftool.

Usage:
    acatome-extract enrich-pdf ./papers/           # Process all PDFs with sidecars
    acatome-extract enrich-pdf ./paper.pdf       # Process single PDF
    acatome-extract enrich-pdf ./ --dry-run      # Preview changes
    acatome-extract enrich-pdf ./ --audit-log ./audit.jsonl

The tool reads metadata from .acatome bundles (and optional .meta.json sidecars),
builds a complete metadata picture, and writes it back to the PDF using exiftool.
It is idempotent: re-running on already-enriched files will skip them unless
--force is used or better metadata is discovered.

Metadata model written to PDF:
    - Title                    → -Title
    - Authors (list)           → -Author (semicolon-joined) + repeated -XMP-dc:Creator
    - DOI                      → -Keywords (with DOI) + -XMP-dc:Identifier="doi:..."
    - Journal/Citation         → -Subject
    - Publisher (if known)     → -XMP-dc:Publisher
    - Year                     → -XMP-dc:Date

DOI workflow (provenance tracked):
    1. Read existing PDF metadata for DOI candidates
    2. Read .acatome bundle header for DOI
    3. Run internal DOI extraction (acatome_meta.pdf.extract_pdf_meta)
    4. Validate via lookup cascade if needed
    5. Optionally use pdf2doi as final fallback
    6. Choose best validated DOI, log conflicts

Exiftool mapping (explicit):
    exiftool -overwrite_original \\
        -Title="Paper Title" \\
        -Author="Author1; Author2; Author3" \\
        -Subject="Journal Name, 2024, 12(5), 234-245" \\
        -Keywords="10.xxxx/yyyy, keyword1, keyword2" \\
        -XMP-dc:Identifier="doi:10.xxxx/yyyy" \\
        -XMP-dc:Creator="Author1" \\
        -XMP-dc:Creator="Author2" \\
        -XMP-dc:Creator="Author3" \\
        -XMP-dc:Publisher="Publisher Name" \\
        -XMP-dc:Date="2024" \\
        file.pdf

Notes on CRC/hash changes:
    Writing metadata to PDFs changes their content hash. To handle this gracefully,
    the .acatome bundle tracks both the original PDF hash and enriched variants:

    - header["pdf_hash"] = original/canonical hash (unchanged, backward compatible)
    - header["pdf_hash_history"] = list of all known-good hashes [original, enriched, ...]

    The watch pipeline validates against any hash in the history, allowing both
    plain and metadata-enriched PDFs to be recognized as the same paper.
    The .acatome bundle remains the canonical source of truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from acatome_meta.lookup import lookup, lookup_doi
from acatome_meta.pdf import extract_doi_from_filename, extract_pdf_meta

from acatome_extract.bundle import read_bundle, update_bundle

log = logging.getLogger(__name__)


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents.

    Args:
        path: Path to file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _update_bundle_hash_history(bundle_path: Path, new_pdf_hash: str) -> bool:
    """Add a new hash to the bundle's pdf_hash_history.

    Maintains backward compatibility: old bundles without pdf_hash_history
    will get one created containing [original_hash, new_hash].

    Args:
        bundle_path: Path to .acatome bundle.
        new_pdf_hash: New hash to add (typically after metadata enrichment).

    Returns:
        True if bundle was updated, False otherwise.
    """
    try:
        bundle = read_bundle(bundle_path)
        header = bundle.get("header", {})

        # Get current history (empty list if not present)
        history: list[str] = header.get("pdf_hash_history", [])

        # If no history yet, seed it with the original pdf_hash
        original_hash = header.get("pdf_hash", "")
        if not history and original_hash:
            history = [original_hash]

        # Add new hash if not already present
        if new_pdf_hash not in history:
            history.append(new_pdf_hash)
            header["pdf_hash_history"] = history
            header["pdf_hash_enriched"] = new_pdf_hash
            bundle["header"] = header
            update_bundle(bundle, bundle_path)
            log.info("Updated %s hash history: %d hashes", bundle_path.name, len(history))
            return True

        return False
    except Exception as e:
        log.warning("Failed to update bundle hash history for %s: %s", bundle_path, e)
        return False


def get_valid_hashes_for_bundle(bundle: dict[str, Any]) -> list[str]:
    """Get all valid PDF hashes for a bundle (original + history).

    This is the validation function used by the watch pipeline.
    Backward compatible: works with old bundles that only have pdf_hash.

    Args:
        bundle: Loaded bundle dict.

    Returns:
        List of valid hash strings (may be empty if no hashes recorded).
    """
    header = bundle.get("header", {})
    hashes: list[str] = []

    # Original hash (backward compatible)
    original = header.get("pdf_hash", "")
    if original:
        hashes.append(original)

    # Hash history (newer bundles)
    history = header.get("pdf_hash_history", [])
    for h in history:
        if h and h not in hashes:
            hashes.append(h)

    return hashes


class DoiProvenance(Enum):
    """Source of a DOI candidate."""

    EXISTING_PDF_METADATA = "existing_pdf_metadata"
    ACATOME_BUNDLE = "acatome_bundle"
    SIDECAR_META = "sidecar_meta"
    INTERNAL_EXTRACTOR = "internal_extractor"
    SECONDARY_VALIDATOR = "secondary_validator"
    PDF2DOI_FALLBACK = "pdf2doi_fallback"
    FILENAME_PATTERN = "filename_pattern"


@dataclass
class DoiCandidate:
    """A DOI candidate with provenance and validation status."""

    doi: str
    provenance: DoiProvenance
    validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize DOI: strip whitespace, lowercase prefix
        self.doi = self.doi.strip().lower()
        if self.doi.startswith("doi:"):
            self.doi = self.doi[4:]


@dataclass
class PdfMetadata:
    """Complete metadata model for a PDF."""

    # Identification
    pdf_path: Path
    pdf_hash: str = ""

    # Core bibliographic metadata
    title: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    doi_provenance: DoiProvenance | None = None
    year: int | None = None
    journal: str = ""
    publisher: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)

    # Source tracking
    acatome_bundle: Path | None = None
    sidecar_meta: dict[str, Any] = field(default_factory=dict)

    # Status
    verified: bool = False
    verify_warnings: list[str] = field(default_factory=list)

    def to_exiftool_args(self) -> list[str]:
        """Build exiftool command-line arguments for this metadata.

        Returns list of arguments like ["-Title=...", "-Author=...", ...]
        """
        args: list[str] = []

        # Basic metadata
        if self.title:
            args.append(f"-Title={self.title}")

        # Author: semicolon-joined for PDF-level display
        if self.authors:
            author_str = "; ".join(self.authors)
            args.append(f"-Author={author_str}")

        # Subject: journal/citation string
        subject_parts: list[str] = []
        if self.journal:
            subject_parts.append(self.journal)
        if self.year:
            subject_parts.append(str(self.year))
        if subject_parts:
            args.append(f"-Subject={', '.join(subject_parts)}")

        # Keywords: DOI first, then other keywords
        kw_list: list[str] = []
        if self.doi:
            kw_list.append(self.doi)
        kw_list.extend(k for k in self.keywords if k != self.doi)
        if kw_list:
            args.append(f"-Keywords={', '.join(kw_list)}")

        # XMP-dc fields (repeated for list values)
        if self.doi:
            args.append(f"-XMP-dc:Identifier=doi:{self.doi}")

        for author in self.authors:
            args.append(f"-XMP-dc:Creator={author}")

        if self.publisher:
            args.append(f"-XMP-dc:Publisher={self.publisher}")

        if self.year:
            args.append(f"-XMP-dc:Date={self.year}")

        if self.title:
            args.append(f"-XMP-dc:Title={self.title}")

        # Add description/abstract if present
        if self.abstract:
            # Truncate very long abstracts for PDF metadata
            short_abstract = (
                self.abstract[:2000] if len(self.abstract) > 2000 else self.abstract
            )
            args.append(f"-XMP-dc:Description={short_abstract}")

        return args

    def get_citation_string(self) -> str:
        """Build a citation-style string for Subject field."""
        parts: list[str] = []
        if self.journal:
            parts.append(self.journal)
        if self.year:
            parts.append(str(self.year))
        return ", ".join(parts)


def _normalize_doi(doi: str) -> str:
    """Normalize DOI: strip prefix, whitespace, lowercase."""
    doi = doi.strip().lower()
    if doi.startswith("doi:"):
        doi = doi[4:]
    return doi


def _is_valid_doi_format(doi: str) -> bool:
    """Check if string looks like a valid DOI format."""
    if not doi:
        return False
    # DOI pattern: 10.{registrant}/{suffix}
    return bool(re.match(r"^10\.\d{4,}/[^\s<>\"}]+$", doi))


def _read_existing_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """Read current metadata from a PDF using exiftool.

    Returns dict with keys like Title, Author, Subject, Keywords, DOI, etc.
    """
    if not shutil.which("exiftool"):
        log.warning("exiftool not found in PATH")
        return {}

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-json",
                "-Title",
                "-Author",
                "-Subject",
                "-Keywords",
                "-Identifier",
                "-Creator",
                "-Publisher",
                "-Date",
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("exiftool failed for %s: %s", pdf_path, result.stderr)
            return {}

        data = json.loads(result.stdout)
        if data and isinstance(data, list):
            return data[0]
        return {}
    except subprocess.TimeoutExpired:
        log.warning("exiftool timeout for %s", pdf_path)
        return {}
    except json.JSONDecodeError as e:
        log.warning("exiftool JSON parse error for %s: %s", pdf_path, e)
        return {}
    except Exception as e:
        log.warning("exiftool error for %s: %s", pdf_path, e)
        return {}


def _extract_doi_candidates(pdf_path: Path) -> list[DoiCandidate]:
    """Extract all possible DOI candidates from a PDF.

    Tries multiple sources and returns a prioritized list.
    """
    candidates: list[DoiCandidate] = []

    # 1. Existing PDF metadata
    existing = _read_existing_pdf_metadata(pdf_path)
    for field_name in ["Identifier", "Keywords", "Subject"]:
        val = existing.get(field_name, "")
        if val and isinstance(val, str):
            # Look for DOI pattern
            match = re.search(r"(?:doi:)?(10\.\d{4,}/[^\s<>\"},]+)", val, re.I)
            if match:
                doi = _normalize_doi(match.group(1))
                if _is_valid_doi_format(doi):
                    candidates.append(
                        DoiCandidate(
                            doi=doi,
                            provenance=DoiProvenance.EXISTING_PDF_METADATA,
                        )
                    )

    # 2. Internal extractor (PyMuPDF-based)
    try:
        pdf_meta = extract_pdf_meta(pdf_path)
        if pdf_meta.get("doi"):
            doi = _normalize_doi(pdf_meta["doi"])
            if _is_valid_doi_format(doi):
                candidates.append(
                    DoiCandidate(
                        doi=doi,
                        provenance=DoiProvenance.INTERNAL_EXTRACTOR,
                    )
                )
    except Exception as e:
        log.debug("Internal DOI extraction failed for %s: %s", pdf_path, e)

    # 3. Filename pattern (Nature, APS, etc.)
    try:
        filename_doi = extract_doi_from_filename(pdf_path)
        if filename_doi:
            doi = _normalize_doi(filename_doi)
            if _is_valid_doi_format(doi):
                candidates.append(
                    DoiCandidate(
                        doi=doi,
                        provenance=DoiProvenance.FILENAME_PATTERN,
                    )
                )
    except Exception as e:
        log.debug("Filename DOI extraction failed for %s: %s", pdf_path, e)

    return candidates


def _try_pdf2doi(pdf_path: Path) -> DoiCandidate | None:
    """Try pdf2doi as a fallback DOI finder.

    Returns None if pdf2doi is not installed or fails.
    """
    try:
        import pdf2doi

        result = pdf2doi.get_identifier(str(pdf_path), trygoogle=False)
        if result and result.get("identifier"):
            doi = _normalize_doi(result["identifier"])
            if _is_valid_doi_format(doi):
                return DoiCandidate(
                    doi=doi,
                    provenance=DoiProvenance.PDF2DOI_FALLBACK,
                )
    except ImportError:
        log.debug("pdf2doi not installed, skipping fallback")
    except Exception as e:
        log.debug("pdf2doi failed for %s: %s", pdf_path, e)
    return None


def _validate_doi(doi: str) -> tuple[bool, dict[str, Any]]:
    """Validate a DOI by looking it up.

    Returns (validated, metadata_dict).
    """
    if not doi:
        return False, {}

    try:
        result = lookup_doi(doi)
        if result and result.get("title"):
            return True, result
    except Exception as e:
        log.debug("DOI validation failed for %s: %s", doi, e)

    return False, {}


def _select_best_doi(
    candidates: list[DoiCandidate],
    use_pdf2doi: bool = False,
    pdf_path: Path | None = None,
) -> DoiCandidate | None:
    """Select the best DOI from candidates.

    Priority:
    1. Already validated candidates (from acatome bundle/lookup)
    2. Validate candidates and pick first that succeeds
    3. If use_pdf2doi and pdf_path provided, try pdf2doi as last resort
    """
    # First, check already-validated candidates
    for c in candidates:
        if c.validated:
            return c

    # Try to validate each candidate in order of provenance trust
    provenance_order = [
        DoiProvenance.ACATOME_BUNDLE,
        DoiProvenance.SIDECAR_META,
        DoiProvenance.SECONDARY_VALIDATOR,
        DoiProvenance.INTERNAL_EXTRACTOR,
        DoiProvenance.EXISTING_PDF_METADATA,
        DoiProvenance.FILENAME_PATTERN,
    ]

    by_provenance: dict[DoiProvenance, list[DoiCandidate]] = {
        p: [] for p in provenance_order
    }
    for c in candidates:
        if c.provenance in by_provenance:
            by_provenance[c.provenance].append(c)

    for prov in provenance_order:
        for c in by_provenance[prov]:
            validated, metadata = _validate_doi(c.doi)
            if validated:
                c.validated = True
                c.metadata = metadata
                return c

    # Try pdf2doi as final fallback
    if use_pdf2doi and pdf_path:
        pdf2doi_candidate = _try_pdf2doi(pdf_path)
        if pdf2doi_candidate:
            validated, metadata = _validate_doi(pdf2doi_candidate.doi)
            if validated:
                pdf2doi_candidate.validated = True
                pdf2doi_candidate.metadata = metadata
            return pdf2doi_candidate

    # Return first candidate even if not validated (best effort)
    for prov in provenance_order:
        if by_provenance[prov]:
            return by_provenance[prov][0]

    return None


def _read_sidecar_meta(pdf_path: Path) -> dict[str, Any]:
    """Read optional .meta.json sidecar alongside PDF."""
    sidecar = pdf_path.with_suffix(".meta.json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _find_acatome_bundle(pdf_path: Path) -> Path | None:
    """Find .acatome bundle corresponding to PDF.

    Looks for .acatome file with same stem in same directory.
    """
    bundle = pdf_path.with_suffix(".acatome")
    if bundle.is_file():
        return bundle

    # Try searching parent directories
    for parent in [pdf_path.parent, pdf_path.parent.parent]:
        if parent.exists():
            for f in parent.glob("*.acatome"):
                # Could match by hash or filename similarity
                if f.stem in pdf_path.stem or pdf_path.stem in f.stem:
                    return f
    return None


def extract_metadata_from_sources(
    pdf_path: Path, use_pdf2doi: bool = False
) -> PdfMetadata:
    """Build complete metadata from all available sources.

    Sources (in order of priority):
    1. .acatome bundle header (highest trust)
    2. .meta.json sidecar
    3. Validated DOI lookup
    4. PDF embedded metadata
    5. Internal extraction
    """
    pdf_path = Path(pdf_path).resolve()

    # Initialize with PDF hash
    pdf_meta = extract_pdf_meta(pdf_path)
    metadata = PdfMetadata(
        pdf_path=pdf_path,
        pdf_hash=pdf_meta.get("pdf_hash", ""),
    )

    # Collect DOI candidates from various sources
    candidates: list[DoiCandidate] = []

    # 1. Check for .acatome bundle
    bundle_path = _find_acatome_bundle(pdf_path)
    if bundle_path:
        metadata.acatome_bundle = bundle_path
        try:
            bundle = read_bundle(bundle_path)
            header = bundle.get("header", {})

            if header.get("doi"):
                candidates.append(
                    DoiCandidate(
                        doi=header["doi"],
                        provenance=DoiProvenance.ACATOME_BUNDLE,
                        validated=True,  # Trust bundle as source of truth
                        metadata=header,
                    )
                )

            # Pre-populate from bundle header
            metadata.title = header.get("title", "")
            metadata.authors = [
                a.get("name", "") for a in header.get("authors", []) if a.get("name")
            ]
            metadata.year = header.get("year")
            metadata.journal = header.get("journal", "")
            metadata.publisher = ""  # Bundle doesn't typically store publisher
            metadata.abstract = header.get("abstract", "")
            metadata.verified = header.get("verified", False)
            metadata.verify_warnings = header.get("verify_warnings", [])
        except Exception as e:
            log.warning("Failed to read bundle %s: %s", bundle_path, e)

    # 2. Check for sidecar meta
    sidecar = _read_sidecar_meta(pdf_path)
    if sidecar:
        metadata.sidecar_meta = sidecar
        if sidecar.get("doi"):
            candidates.append(
                DoiCandidate(
                    doi=sidecar["doi"],
                    provenance=DoiProvenance.SIDECAR_META,
                )
            )
        # Sidecar can override bundle fields
        if sidecar.get("title"):
            metadata.title = sidecar["title"]
        if sidecar.get("author"):
            if isinstance(sidecar["author"], str):
                metadata.authors = [sidecar["author"]]
            elif isinstance(sidecar["author"], list):
                metadata.authors = sidecar["author"]

    # 3. Extract DOI candidates from PDF itself
    pdf_candidates = _extract_doi_candidates(pdf_path)
    candidates.extend(pdf_candidates)

    # Select best DOI
    best_doi = _select_best_doi(candidates, use_pdf2doi=use_pdf2doi, pdf_path=pdf_path)
    if best_doi:
        metadata.doi = best_doi.doi
        metadata.doi_provenance = best_doi.provenance

        # If we got metadata from validation, enrich our record
        if best_doi.metadata:
            if not metadata.title and best_doi.metadata.get("title"):
                metadata.title = best_doi.metadata["title"]
            if not metadata.authors and best_doi.metadata.get("authors"):
                metadata.authors = [
                    a.get("name", "")
                    for a in best_doi.metadata["authors"]
                    if a.get("name")
                ]
            if not metadata.year and best_doi.metadata.get("year"):
                metadata.year = best_doi.metadata["year"]
            if not metadata.journal and best_doi.metadata.get("journal"):
                metadata.journal = best_doi.metadata["journal"]

    # 4. If no bundle metadata, try full lookup cascade
    if not metadata.title:
        try:
            lookup_result = lookup(str(pdf_path))
            if lookup_result.get("title"):
                metadata.title = lookup_result["title"]
            if lookup_result.get("authors"):
                metadata.authors = [
                    a.get("name", "") for a in lookup_result["authors"] if a.get("name")
                ]
            if lookup_result.get("year"):
                metadata.year = lookup_result["year"]
            if lookup_result.get("journal"):
                metadata.journal = lookup_result["journal"]
            if lookup_result.get("doi") and not metadata.doi:
                metadata.doi = lookup_result["doi"]
                metadata.doi_provenance = DoiProvenance.SECONDARY_VALIDATOR
        except Exception as e:
            log.debug("Lookup cascade failed for %s: %s", pdf_path, e)

    # 5. Fallback: embedded PDF metadata
    if not metadata.title:
        info = pdf_meta.get("info", {})
        metadata.title = info.get("title", "")
    if not metadata.authors:
        info = pdf_meta.get("info", {})
        author_str = info.get("author", "")
        if author_str:
            # Split on semicolons or " and "
            if ";" in author_str:
                metadata.authors = [
                    a.strip() for a in author_str.split(";") if a.strip()
                ]
            elif " and " in author_str.lower():
                metadata.authors = [
                    a.strip()
                    for a in re.split(r"\s+and\s+", author_str, flags=re.I)
                    if a.strip()
                ]
            else:
                metadata.authors = [author_str.strip()]

    return metadata


def should_update_file(
    pdf_path: Path, new_metadata: PdfMetadata, force: bool = False
) -> tuple[bool, str]:
    """Determine if PDF should be updated.

    Returns (should_update, reason).
    """
    if force:
        return True, "forced"

    existing = _read_existing_pdf_metadata(pdf_path)

    # Check if we have a DOI to add
    if new_metadata.doi:
        existing_id = existing.get("Identifier", "")
        existing_kw = existing.get("Keywords", "")
        doi_in_id = f"doi:{new_metadata.doi}" in str(existing_id).lower()
        doi_in_kw = new_metadata.doi.lower() in str(existing_kw).lower()
        if not (doi_in_id or doi_in_kw):
            return True, "DOI missing from PDF metadata"

    # Check if we have better title
    if new_metadata.title:
        existing_title = existing.get("Title", "")
        # Only update if existing is empty or looks like garbage
        if not existing_title or len(existing_title) < 10:
            return True, "Title missing or too short"

    # Check if we have author info to add
    if new_metadata.authors:
        existing_author = existing.get("Author", "")
        if not existing_author:
            return True, "Author missing"

    return False, "PDF already has sufficient metadata"


def build_exiftool_command(pdf_path: Path, metadata: PdfMetadata) -> list[str]:
    """Build complete exiftool command as list of arguments."""
    cmd = ["exiftool", "-overwrite_original", "-preserve"]
    cmd.extend(metadata.to_exiftool_args())
    cmd.append(str(pdf_path))
    return cmd


def _backup_pdf(pdf_path: Path) -> Path | None:
    """Create a .bak backup of the PDF before modification.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Path to backup file, or None if backup failed.
    """
    backup_path = pdf_path.with_suffix(pdf_path.suffix + ".bak")
    try:
        import shutil

        shutil.copy2(pdf_path, backup_path)
        log.info("Created backup: %s", backup_path.name)
        return backup_path
    except Exception as e:
        log.warning("Failed to create backup for %s: %s", pdf_path.name, e)
        return None


def write_pdf_metadata(
    pdf_path: Path, metadata: PdfMetadata, dry_run: bool = False
) -> tuple[bool, str]:
    """Write metadata to PDF using exiftool.

    Automatically creates a .bak backup of the original PDF before modification.

    Returns (success, message).
    """
    if not shutil.which("exiftool"):
        return False, "exiftool not found in PATH"

    cmd = build_exiftool_command(pdf_path, metadata)

    if dry_run:
        return True, f"Would run: {' '.join(cmd)}"

    # Create backup before modification
    backup_path = _backup_pdf(pdf_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            msg = result.stdout.strip() or "Metadata written successfully"
            if backup_path:
                msg += f" (backup: {backup_path.name})"
            return True, msg
        else:
            return False, f"exiftool error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "exiftool timeout"
    except Exception as e:
        return False, f"exiftool exception: {e}"


def scan_pdfs(
    target: Path,
    recursive: bool = True,
) -> list[Path]:
    """Scan directory for PDFs that have corresponding .acatome bundles."""
    target = Path(target).resolve()

    if target.is_file():
        if target.suffix.lower() == ".pdf":
            return [target]
        return []

    if not target.is_dir():
        return []

    if recursive:
        pdfs = list(target.rglob("*.pdf"))
    else:
        pdfs = list(target.glob("*.pdf"))

    # Filter to those that have .acatome bundles (safer default)
    # Caller can decide to process all PDFs
    return sorted(pdfs)


@dataclass
class EnrichmentResult:
    """Result of enriching a single PDF."""

    pdf_path: Path
    success: bool
    updated: bool
    message: str
    old_metadata: dict[str, Any] = field(default_factory=dict)
    new_metadata: PdfMetadata | None = None
    error: str | None = None


def enrich_single_pdf(
    pdf_path: Path,
    use_pdf2doi: bool = False,
    force: bool = False,
    dry_run: bool = False,
    require_bundle: bool = True,
    update_bundle_hashes: bool = True,
) -> EnrichmentResult:
    """Enrich a single PDF with metadata.

    This is the main entry point for processing one file.

    Args:
        pdf_path: Path to PDF file.
        use_pdf2doi: Enable pdf2doi fallback.
        force: Force update even if PDF already has metadata.
        dry_run: Preview changes without writing.
        require_bundle: Only process PDFs with .acatome bundles.
        update_bundle_hashes: Update bundle's pdf_hash_history after enrichment.

    Returns:
        EnrichmentResult with status and metadata.
    """
    pdf_path = Path(pdf_path).resolve()
    bundle_path: Path | None = None

    if not pdf_path.exists():
        return EnrichmentResult(
            pdf_path=pdf_path,
            success=False,
            updated=False,
            message="File not found",
            error="File not found",
        )

    # Check for bundle if required
    if require_bundle:
        bundle_path = _find_acatome_bundle(pdf_path)
        if not bundle_path:
            return EnrichmentResult(
                pdf_path=pdf_path,
                success=True,  # Not an error, just skipped
                updated=False,
                message="No .acatome bundle found, skipping",
            )
    else:
        # Try to find bundle even if not required (for hash tracking)
        bundle_path = _find_acatome_bundle(pdf_path)

    # Read old metadata for audit trail
    old_metadata = _read_existing_pdf_metadata(pdf_path)

    # Extract new metadata from all sources
    try:
        new_metadata = extract_metadata_from_sources(pdf_path, use_pdf2doi=use_pdf2doi)
    except Exception as e:
        return EnrichmentResult(
            pdf_path=pdf_path,
            success=False,
            updated=False,
            message=f"Metadata extraction failed: {e}",
            old_metadata=old_metadata,
            error=str(e),
        )

    # Decide if update is needed
    should_update, reason = should_update_file(pdf_path, new_metadata, force=force)

    if not should_update:
        return EnrichmentResult(
            pdf_path=pdf_path,
            success=True,
            updated=False,
            message=f"Skipped: {reason}",
            old_metadata=old_metadata,
            new_metadata=new_metadata,
        )

    # Write metadata
    success, message = write_pdf_metadata(pdf_path, new_metadata, dry_run=dry_run)

    # Update bundle hash history if enrichment succeeded (and not dry run)
    if success and not dry_run and update_bundle_hashes and bundle_path:
        try:
            new_hash = _compute_file_hash(pdf_path)
            _update_bundle_hash_history(bundle_path, new_hash)
        except Exception as e:
            log.warning("Failed to update hash history for %s: %s", pdf_path.name, e)

    return EnrichmentResult(
        pdf_path=pdf_path,
        success=success,
        updated=success and not dry_run,
        message=message,
        old_metadata=old_metadata,
        new_metadata=new_metadata,
        error=message if not success else None,
    )


def enrich_pdfs(
    target: Path,
    use_pdf2doi: bool = False,
    force: bool = False,
    dry_run: bool = False,
    recursive: bool = True,
    require_bundle: bool = True,
    audit_log: Path | None = None,
    progress_callback: callable | None = None,  # type: ignore
) -> list[EnrichmentResult]:
    """Enrich all PDFs in target directory.

    Args:
        target: PDF file or directory to process
        use_pdf2doi: Enable pdf2doi fallback
        force: Force update even if PDF already has metadata
        dry_run: Preview changes without writing
        recursive: Scan subdirectories
        require_bundle: Only process PDFs with .acatome bundles
        audit_log: Path to write audit log (JSONL or CSV)
        progress_callback: Optional callback(current, total, result)

    Returns:
        List of EnrichmentResult for each processed file
    """
    pdfs = scan_pdfs(target, recursive=recursive)

    results: list[EnrichmentResult] = []

    for i, pdf_path in enumerate(pdfs, 1):
        result = enrich_single_pdf(
            pdf_path,
            use_pdf2doi=use_pdf2doi,
            force=force,
            dry_run=dry_run,
            require_bundle=require_bundle,
        )
        results.append(result)

        if progress_callback:
            progress_callback(i, len(pdfs), result)

    # Write audit log if requested
    if audit_log:
        _write_audit_log(audit_log, results)

    return results


def _write_audit_log(log_path: Path, results: list[EnrichmentResult]) -> None:
    """Write audit log in JSONL format."""
    log_path = Path(log_path)

    if str(log_path).lower().endswith(".csv"):
        # CSV format
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "pdf_path",
                    "success",
                    "updated",
                    "old_title",
                    "new_title",
                    "old_doi",
                    "new_doi",
                    "message",
                    "error",
                ]
            )
            for r in results:
                old_doi = ""
                new_doi = r.new_metadata.doi if r.new_metadata else ""
                old_title = r.old_metadata.get("Title", "")
                new_title = r.new_metadata.title if r.new_metadata else ""
                writer.writerow(
                    [
                        datetime.now(UTC).isoformat(),
                        str(r.pdf_path),
                        r.success,
                        r.updated,
                        old_title,
                        new_title,
                        old_doi,
                        new_doi,
                        r.message,
                        r.error or "",
                    ]
                )
    else:
        # JSONL format
        with open(log_path, "w", encoding="utf-8") as f:
            for r in results:
                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "pdf_path": str(r.pdf_path),
                    "success": r.success,
                    "updated": r.updated,
                    "message": r.message,
                    "error": r.error,
                    "old_metadata": r.old_metadata,
                    "new_metadata": {
                        "title": r.new_metadata.title if r.new_metadata else "",
                        "authors": r.new_metadata.authors if r.new_metadata else [],
                        "doi": r.new_metadata.doi if r.new_metadata else "",
                        "doi_provenance": r.new_metadata.doi_provenance.value
                        if r.new_metadata and r.new_metadata.doi_provenance
                        else None,
                        "year": r.new_metadata.year if r.new_metadata else None,
                        "journal": r.new_metadata.journal if r.new_metadata else "",
                    }
                    if r.new_metadata
                    else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_report(results: list[EnrichmentResult]) -> str:
    """Format a human-readable summary of enrichment results."""
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded
    updated = sum(1 for r in results if r.updated)
    skipped = sum(1 for r in results if r.success and not r.updated)

    lines = [
        "PDF Metadata Enrichment Report",
        "==============================",
        f"Total processed: {total}",
        f"Updated: {updated}",
        f"Skipped (already current): {skipped}",
        f"Failed: {failed}",
        "",
    ]

    if failed > 0:
        lines.append("Failures:")
        for r in results:
            if not r.success:
                lines.append(f"  ✗ {r.pdf_path.name}: {r.error or r.message}")
        lines.append("")

    if updated > 0:
        lines.append("Updated files:")
        for r in results:
            if r.updated:
                doi_info = (
                    f" (DOI: {r.new_metadata.doi})"
                    if r.new_metadata and r.new_metadata.doi
                    else ""
                )
                lines.append(f"  ✓ {r.pdf_path.name}{doi_info}")

    return "\n".join(lines)
