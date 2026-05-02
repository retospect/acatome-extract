# Changelog

All notable changes to **acatome-extract** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.6.2] — 2026-04-30

### Added

- `chunker.split_table()` — markdown-table-aware splitter that preserves
  the header row(s) at the top of every output chunk so each chunk stays
  self-describing for retrieval. Falls back gracefully when a "table"
  is actually one corrupted row (no newlines), as can happen with
  multi-page Marker-OCR'd tables.
- `chunker.enforce_hard_max()` — final safety net that force-splits any
  chunk over a hard ceiling (default 16,000 chars) regardless of block
  type. Defends downstream embedders (bge-m3 caps at 8,192 tokens) from
  pathologically long blocks.
- `marker._FTFY_CONFIG` — chemistry-safe `ftfy` configuration applied
  inside `_clean_text()` to repair mojibake (mis-decoded byte sequences
  like `Ã©` → `é`, `â‚‚` → `₂`, `Î±` → `α`) that survived Marker's
  internal passes. Tuned to **never** touch intentional Unicode used
  in scientific text — Greek letters, arrows, sub/superscripts, primes
  (′ ″ for derivatives / minutes), units (°C Å μm ±), or HTML entities
  that chemists legitimately write (`&lt;1 nm`). NFKC normalization,
  `uncurl_quotes`, `unescape_html`, and `fix_character_width` are
  explicitly off; see the inline comment block in `marker.py` for the
  full rationale before flipping any of them.
- New regression suites in `tests/test_marker.py`:
  `TestCleanTextChemistryCorpus` (~50 round-trip parametrizations
  covering Greek / arrows / sub/superscripts / units / math / primes
  + a realistic battery-paper paragraph) and
  `TestCleanTextRepairsMojibake` (~13 parametrizations of the most
  common cp1252-mojibake patterns + idempotency check).

### Changed

- `marker.py` block loop now calls `split_table()` for oversized
  `table` blocks and runs every block through `enforce_hard_max()` as
  a final guard. Previously only `text` and `list` blocks were chunked,
  letting corrupted tables slip through whole and OOM the embedder.
- `_fitz_fallback()` also runs through `enforce_hard_max()` to protect
  against single-word edge cases in `split_text()`.
- `_marker_extract()` now runs the entire `rendered.markdown` blob
  through `_clean_text()` *before* chunking. Previously this path
  skipped the cleanup that the fitz-fallback path performed
  per-page, which meant Marker-extracted papers carried mojibake
  forward into search and embeddings while fitz-extracted ones
  arrived clean. Closes the asymmetry.

### Dependencies

- `ftfy>=6.3` is now a direct dependency. It was already pulled in
  transitively by `marker-pdf`, but the fitz fallback path on installs
  without `[marker]` (e.g. Windows default) needs it pinned directly.

## [0.1.0] — 2026-03-11

### Added

- Initial release.
