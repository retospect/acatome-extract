"""Main extraction pipeline: PDF → .acatome bundle."""

from __future__ import annotations

import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acatome_meta.lookup import lookup
from acatome_meta.pdf import extract_pdf_meta
from acatome_meta.verify import verify_metadata

from acatome_extract.bundle import write_bundle
from acatome_extract.ids import make_node_id, make_paper_id, make_slug
from acatome_extract.marker import extract_blocks_marker

ACATOME_HOME = Path.home() / ".acatome"


def extract(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Extract a single PDF into a .acatome bundle.

    Phase 1 only (structural). No LLM, no embeddings.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Where to write the bundle. Defaults to ~/.acatome/papers/{first_letter}/.
        verify: Whether to verify metadata against PDF text.

    Returns:
        Path to the written .acatome bundle.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: PyMuPDF — metadata & DOI
    pdf_meta = extract_pdf_meta(pdf_path)

    # Step 3: Header metadata (via acatome-meta cascade)
    header = lookup(str(pdf_path))

    # Step 4: Verification
    verify_warnings: list[str] = []
    if not verify:
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
) -> dict[str, list[Path]]:
    """Extract all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Where to write bundles.
        verify: Whether to verify metadata.

    Returns:
        Dict with 'succeeded' and 'failed' lists of paths.
    """
    input_dir = Path(input_dir).resolve()
    pdfs = sorted(input_dir.glob("*.pdf"))

    succeeded: list[Path] = []
    failed: list[Path] = []

    for pdf in pdfs:
        try:
            bundle_path = extract(pdf, output_dir=output_dir, verify=verify)
            succeeded.append(bundle_path)
        except Exception as e:
            failed.append(pdf)
            _write_error(input_dir, pdf, e)

    return {"succeeded": succeeded, "failed": failed}


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
