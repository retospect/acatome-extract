"""Deterministic ID generation for papers and blocks.

``make_slug`` is delegated to :func:`acatome_meta.literature.make_slug` — the
canonical implementation shared across ``acatome-extract``, ``acatome-store``,
and ``precis-mcp``. Re-exported here for backward compatibility.
"""

from __future__ import annotations

from acatome_meta.literature import make_slug

__all__ = ["make_node_id", "make_paper_id", "make_slug"]


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


def make_node_id(paper_id: str, page: int, block_index: int) -> str:
    """Generate deterministic node_id from position."""
    return f"{paper_id}-p{page:02d}-{block_index:03d}"
