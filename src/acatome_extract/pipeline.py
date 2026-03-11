"""Main extraction pipeline: PDF → .acatome bundle."""

from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acatome_meta.lookup import lookup
from acatome_meta.pdf import extract_pdf_meta
from acatome_meta.verify import verify_metadata

from precis_summary import telegram_precis

from acatome_extract.bundle import write_bundle
from acatome_extract.ids import make_node_id, make_paper_id, make_slug
from acatome_extract.marker import extract_blocks_marker

# Block types that skip RAKE summarization (same as enrich skip list)
_SKIP_SUMMARY_TYPES = {"section_header", "title", "author", "equation", "junk"}

ACATOME_HOME = Path.home() / ".acatome"


# Document types that skip metadata lookup (no CrossRef/S2).
_LOCAL_DOC_TYPES = {"datasheet", "manual", "techreport", "notes", "other"}


def extract(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    verify: bool = True,
    doc_type: str = "article",
) -> Path:
    """Extract a single PDF into a .acatome bundle.

    Phase 1 only (structural). No LLM, no embeddings.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Where to write the bundle. Defaults to ~/.acatome/papers/{first_letter}/.
        verify: Whether to verify metadata against PDF text.
        doc_type: Document type — 'article' runs full metadata lookup;
            'datasheet', 'manual', 'techreport', 'notes', 'other' use
            embedded PDF metadata only.

    Returns:
        Path to the written .acatome bundle.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 0: Check for sidecar .meta.json
    sidecar = _read_sidecar(pdf_path)
    if sidecar.get("type"):
        doc_type = sidecar["type"]

    # Step 1: PyMuPDF — metadata & DOI
    pdf_meta = extract_pdf_meta(pdf_path)

    # Step 3: Header metadata
    if doc_type in _LOCAL_DOC_TYPES:
        # Non-article: skip CrossRef/S2 lookup, use embedded PDF metadata
        header = _local_header(pdf_path, pdf_meta, doc_type)
    else:
        header = lookup(str(pdf_path))

    # Apply sidecar overrides (explicit user metadata wins)
    for key in ("title", "year", "doi", "abstract", "journal"):
        if sidecar.get(key):
            header[key] = sidecar[key]
    if sidecar.get("author"):
        # Sidecar author can be string or list of strings
        raw = sidecar["author"]
        if isinstance(raw, str):
            header["authors"] = [{"name": raw}]
        elif isinstance(raw, list):
            header["authors"] = [{"name": a} for a in raw]

    # Override entry_type with the requested doc_type
    header["entry_type"] = doc_type

    # Step 4: Verification
    verify_warnings: list[str] = []
    if not verify or doc_type in _LOCAL_DOC_TYPES:
        verified = True
    elif header.get("first_pages_text"):
        verified, verify_warnings = verify_metadata(header, header["first_pages_text"])
    else:
        # No text to verify against — mark unverified
        verified = False
        verify_warnings = ["no text extracted for verification"]

    # Step 5: Assign IDs
    paper_id = make_paper_id(
        doi=header.get("doi"),
        arxiv_id=header.get("arxiv_id"),
        pdf_hash=pdf_meta["pdf_hash"],
    )
    slug = make_slug(
        header.get("authors", []),
        header.get("year"),
        header.get("title", ""),
    )

    # Step 2: Marker — structured content (falls back to fitz if Marker fails)
    blocks = extract_blocks_marker(pdf_path, paper_id)

    # Step 2½: RAKE summaries (instant, no LLM)
    blocks = _rake_summarize_blocks(blocks)

    # Build bundle
    bundle = _build_bundle(
        paper_id=paper_id,
        slug=slug,
        header=header,
        pdf_meta=pdf_meta,
        blocks=blocks,
        verified=verified,
        verify_warnings=verify_warnings,
    )

    # Write bundle
    if output_dir is None:
        first_letter = slug[0] if slug else "x"
        output_dir = ACATOME_HOME / "papers" / first_letter

    output_dir = Path(output_dir)
    bundle_path = output_dir / f"{slug}.acatome"
    write_bundle(bundle, bundle_path)

    # Copy PDF to papers dir
    pdf_dest = output_dir / f"{slug}.pdf"
    if not pdf_dest.exists():
        shutil.copy2(pdf_path, pdf_dest)

    return bundle_path


def extract_dir(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    verify: bool = True,
    doc_type: str = "article",
) -> dict[str, list[Path]]:
    """Extract all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Where to write bundles.
        verify: Whether to verify metadata.
        doc_type: Document type (see :func:`extract`).

    Returns:
        Dict with 'succeeded' and 'failed' lists of paths.
    """
    input_dir = Path(input_dir).resolve()
    pdfs = sorted(input_dir.glob("*.pdf"))

    succeeded: list[Path] = []
    failed: list[Path] = []

    for pdf in pdfs:
        try:
            bundle_path = extract(pdf, output_dir=output_dir, verify=verify, doc_type=doc_type)
            succeeded.append(bundle_path)
        except Exception as e:
            failed.append(pdf)
            _write_error(input_dir, pdf, e)

    return {"succeeded": succeeded, "failed": failed}


def _read_sidecar(pdf_path: Path) -> dict[str, Any]:
    """Read optional ``<stem>.meta.json`` sidecar alongside a PDF.

    Returns empty dict if no sidecar exists or it's unreadable.
    """
    sidecar_path = pdf_path.with_suffix(".meta.json")
    if not sidecar_path.is_file():
        return {}
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _local_header(
    pdf_path: Path, pdf_meta: dict[str, Any], doc_type: str
) -> dict[str, Any]:
    """Build header from embedded PDF metadata only (no network lookups).

    Used for datasheets, manuals, tech reports, etc.
    """
    info = pdf_meta.get("info", {})
    title = info.get("title", "") or pdf_path.stem.replace("_", " ").replace("-", " ")
    author_str = info.get("author", "")
    authors = [{"name": author_str}] if author_str else []

    # Try to extract year from PDF creation date
    year = None
    creation_date = info.get("creationDate", "")
    if creation_date:
        clean = creation_date.replace("D:", "").strip()
        if len(clean) >= 4 and clean[:4].isdigit():
            yr = int(clean[:4])
            if 1900 <= yr <= 2100:
                year = yr

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": pdf_meta.get("doi"),
        "arxiv_id": None,
        "journal": "",
        "abstract": "",
        "entry_type": doc_type,
        "s2_id": None,
        "source": "local",
        "pdf_hash": pdf_meta["pdf_hash"],
        "page_count": pdf_meta["page_count"],
        "first_pages_text": pdf_meta.get("first_pages_text", ""),
    }


def _build_bundle(
    paper_id: str,
    slug: str,
    header: dict[str, Any],
    pdf_meta: dict[str, Any],
    blocks: list[dict[str, Any]],
    verified: bool,
    verify_warnings: list[str],
) -> dict[str, Any]:
    """Assemble the .acatome bundle dict."""
    now = datetime.now(timezone.utc).isoformat()

    return {
        "header": {
            "paper_id": paper_id,
            "slug": slug,
            "title": header.get("title", ""),
            "authors": header.get("authors", []),
            "year": header.get("year"),
            "doi": header.get("doi"),
            "arxiv_id": header.get("arxiv_id"),
            "journal": header.get("journal", ""),
            "abstract": header.get("abstract", ""),
            "entry_type": header.get("entry_type", "article"),
            "s2_id": header.get("s2_id"),
            "keywords": [],
            "pdf_hash": pdf_meta["pdf_hash"],
            "page_count": pdf_meta["page_count"],
            "source": header.get("source", "embedded"),
            "verified": verified,
            "verify_warnings": verify_warnings,
            "extracted_at": now,
        },
        "blocks": blocks,
        "enrichment_meta": None,
    }


def _rake_summarize_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add RAKE keyword summaries to eligible blocks (instant, no LLM).

    Populates ``summaries["rake"]`` for text blocks with ≥50 chars.
    """
    for block in blocks:
        if block.get("type") in _SKIP_SUMMARY_TYPES:
            continue
        text = block.get("text", "").strip()
        if len(text) < 50:
            continue
        block.setdefault("summaries", {})["rake"] = telegram_precis(text)
    return blocks


def _write_error(input_dir: Path, pdf: Path, error: Exception) -> None:
    """Write error file for failed PDF to ./errors/ subdirectory."""
    errors_dir = input_dir / "errors"
    errors_dir.mkdir(exist_ok=True)
    error_file = errors_dir / f"{pdf.stem}.error.txt"
    error_file.write_text(
        f"PDF: {pdf.name}\n"
        f"Error: {error}\n\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
