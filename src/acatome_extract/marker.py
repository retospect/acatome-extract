"""Structured block extraction via Marker.

Marker v1.x returns MarkdownOutput with:
  .markdown  — full document as markdown string
  .images    — dict[str, PIL.Image] keyed by e.g. '_page_0_Picture_45.jpeg'
  .metadata  — dict with 'table_of_contents' and 'page_stats'

We parse the markdown into structured blocks, classify each, and attach
images from the .images dict.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from acatome_extract.ids import make_node_id

log = logging.getLogger(__name__)

# Ligature normalization map
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


_SPACED_OUT_RE = re.compile(r"(?<![A-Za-z])([A-Za-z](?:\s[A-Za-z]){3,})(?![A-Za-z])")


def _fix_spaced_out(match: re.Match) -> str:
    """Collapse 'M E T H O D S' → 'METHODS'."""
    return match.group(1).replace(" ", "")


def _clean_text(text: str) -> str:
    """Normalize PDF-extracted text.

    - NFC Unicode normalization
    - Replace ligatures (\\ufb01 fi, \\ufb02 fl, etc.)
    - Fix spaced-out kerning artifacts ('M E T H O D S' → 'METHODS')
    - Strip control chars < 0x20 except \\n and \\t
    - Replace \\xa0 (non-breaking space), \\xad (soft hyphen), zero-width chars
    - Collapse multiple blank lines into one
    - Strip trailing whitespace per line
    """
    # NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Ligatures
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)

    # Spaced-out kerning artifacts: "M E T H O D S" → "METHODS"
    text = _SPACED_OUT_RE.sub(_fix_spaced_out, text)

    # Non-breaking space → space, soft hyphen → empty
    text = text.replace("\xa0", " ")
    text = text.replace("\xad", "")

    # Zero-width chars
    text = text.replace("\ufeff", "")  # BOM / zero-width no-break space
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner

    # Strip control chars < 0x20 except \n (0x0a) and \t (0x09)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


# Patterns for classifying markdown blocks
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LIST_RE = re.compile(r"^[-*]\s+")
_DISPLAY_EQ_RE = re.compile(r"^\$\$[\s\S]+\$\$$")
_INLINE_MATH_ONLY_RE = re.compile(r"^\$[^$]+\$$")
_TABLE_RE = re.compile(r"^\|")
_FIGURE_CAPTION_RE = re.compile(
    r"^(?:Fig(?:ure)?\.?\s*\d+)\s*[:\.\-—–]\s*(.+)",
    re.IGNORECASE,
)

# Frontmatter headings that are journal boilerplate, not paper structure.
# Matched case-insensitively against heading text.
_JUNK_HEADING_RE = re.compile(
    r"^(?:"
    r"OPEN\s+ACCESS"
    r"|COPYRIGHT"
    r"|CITATION"
    r"|REVIEWED\s+BY\b"
    r"|EDITED\s+BY\b"
    r"|\*?CORRESPONDENCE\b"
    r"|RECEIVED\s+\d"
    r"|ACCEPTED\s+\d"
    r"|PUBLISHED\s+\d"
    r"|HANDLING\s+EDITOR\b"
    r"|ASSOCIATE\s+EDITOR\b"
    r"|TYPE\s"
    r")",
    re.IGNORECASE,
)

# Markdown link: [text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def extract_blocks_marker(pdf_path: Path, paper_id: str) -> list[dict[str, Any]]:
    """Extract structured blocks from a PDF using Marker.

    Falls back to fitz page-level extraction if Marker fails.
    """
    try:
        return _marker_extract(pdf_path, paper_id)
    except Exception as exc:
        log.warning("Marker failed on %s (%s), using fitz fallback", pdf_path.name, exc)
        return _fitz_fallback(pdf_path, paper_id)


def _patch_text_config_ambiguity() -> None:
    """Monkey-patch transformers 4.48+ get_text_config() ambiguity.

    In transformers >= 4.48, PretrainedConfig.get_text_config() raises
    ValueError when a model config has both 'text_encoder' and 'decoder'
    sub-configs (as surya's models do). Fix: catch the ValueError and
    return text_encoder (or decoder as fallback).
    """
    try:
        from transformers import PretrainedConfig

        _orig = PretrainedConfig.get_text_config

        def _patched(self, **kwargs):
            try:
                return _orig(self, **kwargs)
            except ValueError:
                if hasattr(self, "text_encoder") and self.text_encoder is not None:
                    return self.text_encoder
                if hasattr(self, "decoder") and self.decoder is not None:
                    return self.decoder
                raise

        if PretrainedConfig.get_text_config is not _patched:
            PretrainedConfig.get_text_config = _patched
            log.debug("Patched PretrainedConfig.get_text_config (ambiguity fix)")
    except (ImportError, AttributeError):
        pass


def _patch_surya_config() -> None:
    """Monkey-patch surya SuryaOCRConfig to fix encoder KeyError.

    Bug: __init__ calls super().__init__(**kwargs) which empties kwargs,
    then tries kwargs.pop("encoder"). Fix: pop before super().__init__.
    """
    try:
        from surya.recognition.model.config import SuryaOCRConfig
        from transformers import PretrainedConfig

        _orig = SuryaOCRConfig.__init__

        def _patched_init(self, **kwargs):
            encoder_config = kwargs.pop("encoder", None)
            decoder_config = kwargs.pop("decoder", None)
            PretrainedConfig.__init__(self, **kwargs)
            self.encoder = encoder_config
            self.decoder = decoder_config
            self.is_encoder_decoder = True
            if isinstance(decoder_config, dict):
                self.decoder_start_token_id = decoder_config.get("bos_token_id")
                self.pad_token_id = decoder_config.get("pad_token_id")
                self.eos_token_id = decoder_config.get("eos_token_id")
            elif decoder_config is not None:
                self.decoder_start_token_id = getattr(
                    decoder_config, "bos_token_id", None
                )
                self.pad_token_id = getattr(decoder_config, "pad_token_id", None)
                self.eos_token_id = getattr(decoder_config, "eos_token_id", None)

        # Only patch if the bug exists (no default for encoder kwarg)
        import inspect

        sig = inspect.signature(_orig)
        if "encoder" not in sig.parameters:
            SuryaOCRConfig.__init__ = _patched_init
            log.debug("Patched SuryaOCRConfig.__init__ (encoder KeyError fix)")
    except ImportError:
        pass


def _marker_extract(pdf_path: Path, paper_id: str) -> list[dict[str, Any]]:
    """Run Marker and parse its MarkdownOutput into block schema."""
    import warnings

    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated")

    _patch_text_config_ambiguity()
    _patch_surya_config()
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))

    md = rendered.markdown
    images = rendered.images or {}
    metadata = rendered.metadata or {}

    # Build page boundary map from metadata
    toc = metadata.get("table_of_contents", [])
    page_stats = metadata.get("page_stats", [])
    total_pages = len(page_stats) if page_stats else 1

    # Split markdown into raw blocks (double newline separated)
    raw_chunks = re.split(r"\n{2,}", md)

    blocks: list[dict[str, Any]] = []
    current_section: list[str] = []
    block_counts: dict[int, int] = {}

    # Estimate page assignment: distribute chunks across pages
    page_assignments = _assign_pages(raw_chunks, total_pages, toc)

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        page_num = page_assignments[i] if i < len(page_assignments) else 0
        block_type, text = _classify_chunk(chunk)

        if block_type == "section_header":
            # Strip markdown links from heading text
            text = _MD_LINK_RE.sub(r"\1", text).strip()
            current_section = [text]
            # Still emit the heading as a block
        elif block_type == "skip":
            continue

        if page_num not in block_counts:
            block_counts[page_num] = 0
        idx = block_counts[page_num]
        block_counts[page_num] = idx + 1

        block: dict[str, Any] = {
            "node_id": make_node_id(paper_id, page_num, idx),
            "page": page_num,
            "type": block_type,
            "text": text,
            "section_path": list(current_section),
            "bbox": None,
            "embeddings": {},
            "summary": None,
        }

        # Attach images referenced in this chunk
        img_refs = _IMAGE_RE.findall(chunk)
        for _alt, ref in img_refs:
            img_key = _find_image_key(ref, images)
            if img_key:
                b64, mime = _encode_pil_image(images[img_key])
                block["image_base64"] = b64
                block["image_mime"] = mime
                block["type"] = "figure"
                break

        blocks.append(block)

    # Match figure captions
    blocks = _match_captions(blocks)

    # Mark frontmatter junk
    blocks = _mark_junk(blocks)

    return blocks


def _classify_chunk(chunk: str) -> tuple[str, str]:
    """Classify a markdown chunk and return (block_type, cleaned_text)."""
    first_line = chunk.split("\n")[0].strip()

    # Heading
    m = _HEADING_RE.match(first_line)
    if m:
        return "section_header", m.group(2).strip()

    # Image-only block
    if _IMAGE_RE.match(first_line) and len(chunk.split("\n")) <= 2:
        alt = _IMAGE_RE.match(first_line).group(1)
        return "figure", alt or ""

    # Table
    lines = chunk.strip().split("\n")
    if len(lines) >= 2 and all(_TABLE_RE.match(l.strip()) for l in lines):
        return "table", chunk

    # Display equation ($$...$$)
    stripped = chunk.strip()
    if _DISPLAY_EQ_RE.match(stripped):
        content = stripped.strip("$").strip()
        # Skip short formula fragments (e.g. "$$R^{3}$$")
        if len(content) < 40:
            return "skip", ""
        return "equation", content

    # Skip tiny inline-math-only fragments (e.g. "$R^{3}$")
    if _INLINE_MATH_ONLY_RE.match(stripped) and len(stripped) < 80:
        return "skip", ""

    # List block (all lines start with - or *)
    if all(_LIST_RE.match(l.strip()) for l in lines if l.strip()):
        return "list", chunk

    # Default text
    return "text", chunk


def _assign_pages(chunks: list[str], total_pages: int, toc: list[dict]) -> list[int]:
    """Assign each chunk to a page number.

    Uses TOC entries with page_ids as anchors. Between anchors,
    chunks are assigned to the most recent page.
    """
    if total_pages <= 1:
        return [0] * len(chunks)

    # Build anchor map: chunk_index → page_id from TOC title matches
    anchors: dict[int, int] = {}
    for entry in toc:
        title = (entry.get("title") or "").replace("\n", " ").strip()
        page_id = entry.get("page_id", 0)
        if not title:
            continue
        # Find chunk containing this title
        for i, chunk in enumerate(chunks):
            if title[:40] in chunk.replace("\n", " "):
                anchors[i] = page_id
                break

    # Assign pages: propagate from anchors
    assignments = [0] * len(chunks)
    current_page = 0
    for i in range(len(chunks)):
        if i in anchors:
            current_page = anchors[i]
        assignments[i] = min(current_page, total_pages - 1)

    return assignments


def _find_image_key(ref: str, images: dict) -> str | None:
    """Find matching image key from a markdown image reference."""
    # Direct match
    if ref in images:
        return ref
    # Try matching by filename portion
    ref_name = ref.rsplit("/", 1)[-1]
    for key in images:
        if ref_name in key or key in ref_name:
            return key
    return None


def _encode_pil_image(img: Any) -> tuple[str, str]:
    """Encode a PIL Image to base64 PNG."""
    buf = io.BytesIO()
    fmt = "PNG"
    mime = "image/png"
    # Use JPEG for larger images
    if hasattr(img, "size"):
        w, h = img.size
        if w * h > 500_000:
            fmt = "JPEG"
            mime = "image/jpeg"
            if img.mode == "RGBA":
                img = img.convert("RGB")
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, mime


def _match_captions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match figure blocks with captions from adjacent text blocks."""
    result = []
    skip_next = False

    for i, block in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue

        if block.get("type") == "figure" and i + 1 < len(blocks):
            next_block = blocks[i + 1]
            next_text = next_block.get("text", "")
            if _FIGURE_CAPTION_RE.match(next_text.strip()):
                block["text"] = next_text.strip()
                skip_next = True

        result.append(block)

    return result


def _mark_junk(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Demote frontmatter boilerplate blocks to type 'junk'.

    A junk heading and all blocks that follow it (inheriting its
    section_path) are marked as junk.  Once a real section_header
    appears, junk mode ends and subsequent blocks are normal.
    """
    in_junk = False
    junk_section: list[str] | None = None

    for block in blocks:
        btype = block.get("type", "")
        text = block.get("text", "")

        if btype == "section_header":
            if _JUNK_HEADING_RE.match(text):
                in_junk = True
                junk_section = block.get("section_path")
                block["type"] = "junk"
            else:
                in_junk = False
                junk_section = None
        elif in_junk:
            # Followers of a junk heading inherit the junk section_path
            if block.get("section_path") == junk_section:
                block["type"] = "junk"

    return blocks


_HEADING_PATTERN = re.compile(r"^(\d+[\.\s]\s*\S|[A-Z][A-Z\s]{2,}$)", re.MULTILINE)


def _fitz_fallback(pdf_path: Path, paper_id: str) -> list[dict[str, Any]]:
    """Fallback: extract and chunk text via fitz + recursive splitter.

    1. Extract full page text via ``page.get_text()``
    2. Strip repeating headers/footers across pages
    3. Chunk each page's text with ``acatome_extract.chunker``
    4. Classify chunks (heading detection)
    """
    import fitz

    from acatome_extract.chunker import split_text

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count

    # Collect raw page texts
    page_texts: list[tuple[int, str]] = []
    for page_num in range(total_pages):
        text = _clean_text(doc[page_num].get_text())
        if text:
            page_texts.append((page_num, text))
    doc.close()

    # Strip repeating headers/footers before chunking
    page_texts = _strip_running_lines(page_texts, total_pages)

    # Chunk each page and build blocks
    blocks: list[dict[str, Any]] = []
    current_section: list[str] = []

    for page_num, text in page_texts:
        chunks = split_text(text)
        for idx, chunk in enumerate(chunks):
            block_type = "text"
            if _is_likely_heading(chunk):
                block_type = "section_header"
                current_section = [chunk]

            blocks.append(
                {
                    "node_id": make_node_id(paper_id, page_num, idx),
                    "page": page_num,
                    "type": block_type,
                    "text": chunk,
                    "section_path": list(current_section),
                    "bbox": None,
                    "embeddings": {},
                    "summary": None,
                }
            )

    log.info(
        "fitz fallback: %d pages → %d chunks",
        total_pages,
        len(blocks),
    )
    return blocks


def _strip_running_lines(
    page_texts: list[tuple[int, str]], total_pages: int
) -> list[tuple[int, str]]:
    """Remove lines that repeat verbatim on ≥40% of pages (headers/footers).

    Works on the first and last 3 lines of each page.
    """
    if total_pages < 3:
        return page_texts

    threshold = total_pages * 0.4
    line_pages: dict[str, set[int]] = {}

    for page_num, text in page_texts:
        lines = text.split("\n")
        candidates = lines[:3] + lines[-3:]
        for line in candidates:
            line = line.strip()
            if 3 < len(line) <= 120:
                line_pages.setdefault(line, set()).add(page_num)

    repeating = {ln for ln, pages in line_pages.items() if len(pages) >= threshold}
    if not repeating:
        return page_texts

    log.debug("Stripping %d repeating header/footer lines", len(repeating))
    result: list[tuple[int, str]] = []
    for page_num, text in page_texts:
        lines = [ln for ln in text.split("\n") if ln.strip() not in repeating]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            result.append((page_num, cleaned))
    return result


def _is_likely_heading(text: str) -> bool:
    """Heuristic: short single-line text that looks like a section heading."""
    if "\n" in text or len(text) > 120 or len(text) < 3:
        return False
    # ALL CAPS (but not just an abbreviation)
    if text == text.upper() and len(text) > 5 and not text.endswith("."):
        return True
    # Numbered heading: "1. Introduction", "2 Methods", "3.1 Results"
    if re.match(r"^\d+[\.\d]*[\.\s]\s*[A-Z]", text) and not text.endswith("."):
        return True
    return False
