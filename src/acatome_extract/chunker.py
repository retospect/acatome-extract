"""Recursive character text splitter for document chunking.

Splits text into chunks of roughly ``chunk_size`` characters, preferring
to break at natural boundaries (paragraphs → newlines → sentences → words).
Adjacent chunks overlap by ``chunk_overlap`` characters to preserve context
across chunk boundaries.
"""

from __future__ import annotations

# Default separators, tried in order (prefer paragraph → line → sentence → word)
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", ", ", " "]

# Reasonable defaults for academic papers
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* characters.

    The algorithm tries each separator in order.  For the first separator
    that produces pieces, it keeps pieces that fit and recursively splits
    those that don't (using the remaining separators).  Adjacent chunks
    share *chunk_overlap* characters of context.

    Returns a list of non-empty strings, each ≤ ``chunk_size`` chars
    (unless a single word exceeds the limit, in which case it is kept
    whole to avoid mid-word splits).
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    seps = separators if separators is not None else list(DEFAULT_SEPARATORS)

    return _recursive_split(text, chunk_size, chunk_overlap, seps)


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Core recursive splitting logic."""
    # Base case: text fits
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Try each separator
    for i, sep in enumerate(separators):
        pieces = _split_keeping_sep(text, sep)
        if len(pieces) <= 1:
            continue  # separator not found; try next

        # Merge small pieces back together up to chunk_size
        merged = _merge_pieces(pieces, chunk_size, chunk_overlap, sep)

        # Recursively split any chunk that's still too big
        remaining_seps = separators[i + 1 :]
        result: list[str] = []
        for chunk in merged:
            if len(chunk) <= chunk_size:
                stripped = chunk.strip()
                if stripped:
                    result.append(stripped)
            elif remaining_seps:
                result.extend(
                    _recursive_split(chunk, chunk_size, chunk_overlap, remaining_seps)
                )
            else:
                # No more separators — keep as-is (won't split mid-word)
                stripped = chunk.strip()
                if stripped:
                    result.append(stripped)
        return result

    # No separator worked — return text as-is
    stripped = text.strip()
    return [stripped] if stripped else []


def _split_keeping_sep(text: str, sep: str) -> list[str]:
    """Split text by *sep*, keeping the separator at the start of each piece
    (except the first)."""
    parts = text.split(sep)
    if len(parts) <= 1:
        return parts

    result = [parts[0]]
    for part in parts[1:]:
        result.append(sep + part)
    return result


def _merge_pieces(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
    sep: str,
) -> list[str]:
    """Greedily merge adjacent pieces into chunks up to *chunk_size*.

    When starting a new chunk, includes up to *chunk_overlap* characters
    from the tail of the previous chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece)

        if current and current_len + piece_len > chunk_size:
            # Flush current buffer
            chunk_text = "".join(current)
            if chunk_text.strip():
                chunks.append(chunk_text)

            # Build overlap from end of current buffer
            overlap_pieces: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) > chunk_overlap:
                    break
                overlap_pieces.insert(0, prev)
                overlap_len += len(prev)

            current = overlap_pieces
            current_len = overlap_len

        current.append(piece)
        current_len += piece_len

    # Flush remaining
    if current:
        chunk_text = "".join(current)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks
