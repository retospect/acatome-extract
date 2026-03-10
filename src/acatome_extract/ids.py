"""Deterministic ID generation for papers and blocks."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {"a", "an", "the", "new", "on", "of", "in", "for", "to", "and", "with"}


def make_paper_id(
    doi: str | None = None,
    arxiv_id: str | None = None,
    pdf_hash: str | None = None,
) -> str:
    """Generate deterministic paper_id. Priority: arxiv > doi > sha256.

    Args:
        doi: DOI string (without prefix).
        arxiv_id: arXiv identifier.
        pdf_hash: SHA256 hash of PDF content.

    Returns:
        Paper ID string like 'arxiv:2401.12345' or 'doi:10.1038/...'
    """
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi}"
    if pdf_hash:
        return f"sha256:{pdf_hash[:12]}"
    raise ValueError("At least one of doi, arxiv_id, or pdf_hash is required")


def make_slug(authors: list[dict], year: int | None, title: str) -> str:
    """Generate human-readable slug: {surname}{year}{keyword}.

    Rules: ASCII-only, lowercase, skip stopwords, 'anon' if no author,
    '0000' if no year.  Non-Latin titles (Chinese, Arabic, etc.) get a
    short hash as keyword.
    """
    # First author surname
    if authors:
        name = authors[0].get("name", "")
        surname = name.split(",")[0].strip().lower() if name else "anon"
    else:
        surname = "anon"

    # ASCII-only (transliterate accented → base letter)
    surname = unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode()
    surname = re.sub(r"[^a-z]", "", surname)
    if not surname:
        surname = "anon"

    # Year
    yr = str(year) if year else "0000"

    # First content word from title
    # Transliterate first, then extract ASCII words
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z]+", ascii_title.lower())
    keyword = next(
        (w for w in words if w not in STOPWORDS), words[0] if words else ""
    )

    # Non-Latin titles may yield no ASCII words — use short hash
    if not keyword:
        if title.strip():
            import hashlib

            keyword = hashlib.sha256(title.encode()).hexdigest()[:6]
        else:
            keyword = "untitled"

    return f"{surname}{yr}{keyword}"


def make_node_id(paper_id: str, page: int, block_index: int) -> str:
    """Generate deterministic node_id from position."""
    return f"{paper_id}-p{page:02d}-{block_index:03d}"
