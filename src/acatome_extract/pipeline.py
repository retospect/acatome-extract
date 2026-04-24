"""Main extraction pipeline: PDF → .acatome bundle."""

from __future__ import annotations

import json
import logging
import re
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acatome_meta.lookup import lookup
from acatome_meta.pdf import extract_pdf_meta, is_garbage_title
from acatome_meta.verify import verify_metadata
from precis_summary import telegram_precis

from acatome_extract.bundle import write_bundle
from acatome_extract.ids import make_paper_id, make_slug
from acatome_extract.marker import extract_blocks_marker

log = logging.getLogger(__name__)

# Block types that skip RAKE summarization (same as enrich skip list)
_SKIP_SUMMARY_TYPES = {"section_header", "title", "author", "equation", "junk"}

ACATOME_HOME = Path.home() / ".acatome"


# Document types that skip metadata lookup (no CrossRef/S2).
_LOCAL_DOC_TYPES = {"datasheet", "manual", "techreport", "notes", "other"}

# Keys a sidecar .meta.json can override on the looked-up header.
# Explicit None clears the field (backward-compat: empty string is ignored).
_SIDECAR_OVERRIDE_KEYS = (
    "title",
    "year",
    "doi",
    "abstract",
    "journal",
    "s2_id",
    "arxiv_id",
)


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

    # Apply sidecar overrides (explicit user metadata wins).
    # Explicit None clears a field — useful when the initial Crossref/S2
    # lookup returned a wrong s2_id/arxiv_id that would otherwise contaminate
    # downstream dedup. Empty strings are ignored (treated as "not set").
    for key in _SIDECAR_OVERRIDE_KEYS:
        if key not in sidecar:
            continue
        val = sidecar[key]
        if val == "":
            continue
        header[key] = val  # None clears, non-empty overrides
    if "author" in sidecar:
        raw = sidecar["author"]
        if isinstance(raw, str) and raw:
            header["authors"] = [{"name": raw}]
        elif isinstance(raw, list):
            header["authors"] = [{"name": a} for a in raw if a]
        elif raw is None:
            header["authors"] = []

    # Override entry_type with the requested doc_type
    header["entry_type"] = doc_type

    # Step 4: Verification
    verify_warnings: list[str] = []
    if not verify or doc_type in _LOCAL_DOC_TYPES:
        verified = True
    elif "verified" in sidecar:
        # Sidecar explicitly attests to metadata correctness — trust the user
        # (they edited this file by hand after reviewing the PDF).
        verified = bool(sidecar["verified"])
        if not verified:
            verify_warnings = ["sidecar marked unverified"]
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

    # Step 6: Rescue metadata from blocks if lookup returned garbage.
    #
    # Triggers when:
    #   - title missing entirely, OR
    #   - title matches a known-bad embedded-metadata pattern (InDesign
    #     filenames, manuscript tracking IDs, LaTeX template boilerplate —
    #     see :func:`is_garbage_title`). This catches cases where CrossRef
    #     itself returned garbage for a valid DOI (e.g. APS's revtex
    #     boilerplate leaking as the title), OR
    #   - authors list is empty.
    #
    # The rescue function gates on ``header.get("title")`` being empty,
    # so we clear a garbage title here before calling it.
    title = header.get("title", "") or ""
    if is_garbage_title(title):
        log.info("clearing garbage title from lookup: %r", title[:60])
        header["title"] = ""
        title = ""
    if not title or not header.get("authors"):
        rescued = _rescue_metadata_from_blocks(blocks, header)
        if rescued:
            header.update(rescued)
            slug = make_slug(
                header.get("authors", []),
                header.get("year"),
                header.get("title", ""),
            )
            log.info("rescued metadata from text: slug=%s title=%r", slug, header.get("title", "")[:60])
            # Re-verify: the title/authors we now have came from block text,
            # so they should fuzz-match the first-pages text by construction.
            # Without this, ``verified`` stays False (computed from the pre-
            # rescue garbage title) and the paper would be rejected later.
            if (
                verify
                and doc_type not in _LOCAL_DOC_TYPES
                and "verified" not in sidecar
                and header.get("first_pages_text")
            ):
                verified, verify_warnings = verify_metadata(
                    header, header["first_pages_text"]
                )

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
            bundle_path = extract(
                pdf, output_dir=output_dir, verify=verify, doc_type=doc_type
            )
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


# Regex: line that looks like an author list (comma/and-separated names,
# possibly with superscripts, affiliations stripped).
_AUTHOR_LINE_RE = re.compile(
    r"^[A-Z\u00C0-\u024F][\w\s.\-\u00C0-\u024F]+(?:,\s*[A-Z\u00C0-\u024F][\w\s.\-\u00C0-\u024F]+)*$"
)


def _rescue_metadata_from_blocks(
    blocks: list[dict[str, Any]],
    header: dict[str, Any],
) -> dict[str, Any] | None:
    """Try to extract title and authors from the first few text blocks.

    Called when the metadata lookup cascade failed to produce usable
    metadata (no title or no authors).  Marker usually puts the title
    as the first text block and the author list as the second.

    Returns a dict of rescued fields to merge into *header*, or None.
    """
    # Collect first few substantial text blocks (skip junk, headers, tiny)
    candidates: list[str] = []
    for block in blocks[:15]:
        btype = block.get("type", "")
        if btype in ("junk", "equation", "figure", "table"):
            continue
        text = block.get("text", "").strip()
        if not text or len(text) < 5:
            continue
        candidates.append(text)
        if len(candidates) >= 5:
            break

    if not candidates:
        return None

    rescued: dict[str, Any] = {}

    # Title: first candidate that's short enough to be a title (< 300 chars)
    # and doesn't look like a section header number
    if not header.get("title"):
        for cand in candidates:
            # Skip section headers like "1. Introduction"
            if re.match(r"^\d+[\.\s]", cand):
                continue
            # Skip "Abstract" headers
            if cand.lower().strip() in ("abstract", "contents", "references"):
                continue

            if len(cand) <= 300:
                rescued["title"] = cand
                break

            # For long blocks: title+author often merged by Marker.
            # Split on newlines and extract the title from the first lines.
            title, remaining_authors = _split_title_from_block(cand)
            if title:
                rescued["title"] = title
                if remaining_authors and not header.get("authors"):
                    rescued["authors"] = remaining_authors
                break

    # Authors: look for a block after the title that contains names
    # (skip if already rescued from a merged block above)
    if not header.get("authors") and not rescued.get("authors"):
        title_text = rescued.get("title") or header.get("title", "")
        for cand in candidates:
            if cand == title_text:
                continue
            # Skip if it's clearly abstract/body text (has sentences)
            if len(cand) > 500 or cand.count(". ") > 3:
                continue
            # Skip known non-author patterns
            low = cand.lower()
            if low.startswith("abstract") or low.startswith("keyword"):
                continue
            authors = _parse_author_block(cand)
            if authors:
                rescued["authors"] = authors
                break

    # Try S2 title lookup with rescued title
    if rescued.get("title") and not header.get("doi"):
        try:
            import os

            from acatome_meta.semantic_scholar import lookup_s2

            s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
            s2_result = lookup_s2(rescued["title"], api_key=s2_key)
            if s2_result and s2_result.get("title"):
                # Year-sanity check. S2's fuzzy title search can return a
                # plausible-but-wrong paper when the rescued title is a
                # generic phrase ("The rise of graphene" matches a 2024
                # review whose subtitle contains it). If S2's year differs
                # from the PDF year by more than 2, reject the hit — the
                # correct outcome is to leave rescued as text-only so the
                # verify gate later rejects it instead of us accepting bad
                # metadata. 2-year tolerance covers preprint→journal lag.
                pdf_year = header.get("year")
                s2_year = s2_result.get("year")
                if (
                    pdf_year
                    and s2_year
                    and abs(int(s2_year) - int(pdf_year)) > 2
                ):
                    log.warning(
                        "rejecting S2 rescue hit %r: year mismatch "
                        "(pdf=%s vs s2=%s)",
                        s2_result.get("title", "")[:60],
                        pdf_year,
                        s2_year,
                    )
                else:
                    log.info("S2 rescue hit: %s", s2_result.get("title", "")[:60])
                    # Merge S2 result (higher quality) into rescued
                    for key in ("title", "authors", "year", "doi", "arxiv_id",
                                "journal", "abstract", "s2_id"):
                        if s2_result.get(key):
                            rescued[key] = s2_result[key]
                    rescued["source"] = "s2_rescue"
        except Exception as exc:
            log.debug("S2 rescue lookup failed: %s", exc)

    if not rescued:
        return None

    rescued.setdefault("source", "text_rescue")
    log.info("rescued %d field(s) from block text", len(rescued) - 1)
    return rescued


def _split_title_from_block(text: str) -> tuple[str | None, list[dict[str, str]] | None]:
    """Split a merged title+author block into (title, authors).

    Marker sometimes puts title, author names, and abstract into one block.
    Strategy: split on newlines, collect title lines (short, no commas/numbers
    suggesting author affiliations), stop when we hit an author-like line.
    """
    lines = text.split("\n")

    # Strip leading arXiv header lines (e.g. "arXiv:2301.12345v3  [cs.LG]  29 Jan 2023")
    while lines and re.match(r"^arXiv:\d", lines[0].strip()):
        lines = lines[1:]

    if not lines:
        return None, None

    title_lines: list[str] = []
    author_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if title_lines:
                author_start = i + 1
                break
            continue

        # Stop at Abstract/Contents/date lines
        low = stripped.lower()
        if low.startswith("abstract") or low.startswith("contents"):
            author_start = i
            break

        # Stop at date-like lines (e.g. "30th October 2018")
        if re.match(r"^\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}", stripped):
            author_start = i
            break

        # Author-like line: contains superscript markers, multiple commas + capitalized names
        has_affiliation_markers = bool(re.search(r"[†‡§¶∗⊥#]|\d+[,∗]", stripped))
        has_multi_names = stripped.count(",") >= 2 and re.search(r"[A-Z]\.\s", stripped)
        if has_affiliation_markers or has_multi_names:
            if title_lines:
                author_start = i
                break

        # If it looks like continuation of a title (no period at end, reasonable length)
        if len(stripped) < 200 and not stripped.endswith("."):
            title_lines.append(stripped)
        else:
            if title_lines:
                author_start = i
                break
            title_lines.append(stripped)

        # Cap title at 3 lines
        if len(title_lines) >= 3:
            author_start = i + 1
            break

    if not title_lines:
        return None, None

    title = " ".join(title_lines)
    if len(title) > 300:
        title = title[:300]

    # Try to parse authors from remaining lines
    authors = None
    if author_start is not None and author_start < len(lines):
        author_text = "\n".join(lines[author_start:author_start + 3])
        authors = _parse_author_block(author_text)

    return title, authors


def _parse_author_block(text: str) -> list[dict[str, str]] | None:
    """Parse an author block into a list of {name: ...} dicts.

    Strips superscript markers, affiliation numbers, and common symbols.
    Returns None if no plausible author names found.
    """
    # Strip superscript markers and affiliations
    clean = re.sub(r"<sup>[^<]*</sup>", "", text)
    clean = re.sub(r"\*+", "", clean)
    clean = re.sub(r"[†‡§¶∗⊥#]", "", clean)
    # Strip affiliation numbers attached to names (e.g. "Author1,2")
    clean = re.sub(r"(\w)\d+(?:,\d+)*", r"\1", clean)
    # Strip standalone numbers
    clean = re.sub(r"\b\d+\b", "", clean)
    # Strip email-like patterns
    clean = re.sub(r"\S+@\S+", "", clean)
    # Strip leading "and"
    clean = re.sub(r"^\s*and\s+", "", clean, flags=re.IGNORECASE)

    # First line only (remaining lines are usually affiliations)
    first_line = clean.split("\n")[0].strip()
    if not first_line:
        return None

    # Split on comma, semicolon, " and "
    parts = re.split(r"[;,]|\band\b", first_line)
    names = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    # Must have at least one name-like token (capitalized word)
    name_like = [n for n in names if re.match(r"[A-Z\u00C0-\u024F]", n)]
    if name_like:
        return [{"name": n} for n in name_like[:10]]
    return None


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
    now = datetime.now(UTC).isoformat()

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
        f"PDF: {pdf.name}\nError: {error}\n\nTraceback:\n{traceback.format_exc()}"
    )
