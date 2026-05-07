# PDF Metadata Enrichment Tool

## Overview

The `enrich-pdf` command writes DOI, title, authors, and other metadata back into PDF files using exiftool. It sources metadata from `.acatome` bundles, `.meta.json` sidecars, and internal DOI extraction/validation.

## Installation

```bash
# Basic install (exiftool required)
pip install acatome-extract

# With pdf2doi fallback support
pip install acatome-extract[pdf2doi]
```

**System dependency:** Install exiftool via your package manager:
- macOS: `brew install exiftool`
- Ubuntu/Debian: `apt-get install libimage-exiftool-perl`

## Usage

### Basic Examples

```bash
# Process all PDFs with .acatome bundles in a directory
acatome-extract enrich-pdf ./papers/

# Process single PDF
acatome-extract enrich-pdf ./paper.pdf

# Preview changes (dry run)
acatome-extract enrich-pdf ./papers/ --dry-run

# Process all PDFs (not just those with bundles)
acatome-extract enrich-pdf ./ --no-require-bundle

# Force update even if PDF already has metadata
acatome-extract enrich-pdf ./papers/ --force

# Enable pdf2doi fallback for stubborn PDFs
acatome-extract enrich-pdf ./papers/ --use-pdf2doi

# Write audit log for tracking changes
acatome-extract enrich-pdf ./papers/ --audit-log ./audit.jsonl

# Verbose progress output
acatome-extract enrich-pdf ./papers/ --verbose
```

## Metadata Model

The tool writes the following metadata to each PDF:

| Field | PDF Tag | XMP-dc Tag | Source |
|-------|---------|------------|--------|
| Title | `-Title` | `-XMP-dc:Title` | .acatome bundle → sidecar → lookup |
| Authors (joined) | `-Author` | - | .acatome bundle → sidecar → PDF meta |
| Authors (individual) | - | `-XMP-dc:Creator` (repeated) | .acatome bundle |
| DOI | `-Keywords` | `-XMP-dc:Identifier` | bundle → validation → extraction |
| Journal/Year | `-Subject` | `-XMP-dc:Date` | .acatome bundle → lookup |
| Publisher | - | `-XMP-dc:Publisher` | lookup |
| Abstract | - | `-XMP-dc:Description` | .acatome bundle (truncated) |

## DOI Workflow

The DOI extraction cascade (in order of priority):

1. **.acatome bundle** (`acatome_bundle`) — trusted source of truth
2. **.meta.json sidecar** (`sidecar_meta`) — user override
3. **Existing PDF metadata** (`existing_pdf_metadata`)
4. **Internal extractor** (`internal_extractor`) — PyMuPDF-based extraction
5. **Filename pattern** (`filename_pattern`) — Nature, APS patterns
6. **Secondary validator** (`secondary_validator`) — CrossRef lookup
7. **pdf2doi fallback** (`pdf2doi_fallback`) — optional external tool

## Exiftool Mapping

The tool generates exiftool commands equivalent to:

```bash
exiftool -overwrite_original \
  -Title="Synthesis of MOF-5 using zinc nitrate precursor" \
  -Author="Smith, J.; Jones, A.; Lee, K." \
  -Subject="Journal of Materials Chemistry A, 2024, 12(5), 234-245" \
  -Keywords="10.1039/d4ta00123k, MOF-5, DFT, VASP, synthesis" \
  -XMP-dc:Identifier="doi:10.1039/d4ta00123k" \
  -XMP-dc:Creator="Smith, J." \
  -XMP-dc:Creator="Jones, A." \
  -XMP-dc:Creator="Lee, K." \
  -XMP-dc:Publisher="Royal Society of Chemistry" \
  -XMP-dc:Date="2024" \
  -XMP-dc:Description="Abstract text..." \
  file.pdf
```

## Idempotency

Re-running the tool is safe:
- PDFs with sufficient metadata are skipped
- Use `--force` to re-write anyway
- Use `--dry-run` to preview without changes

## Audit Logging

JSONL format (default):
```json
{"timestamp": "2024-01-15T10:30:00Z", "pdf_path": "/path/to/paper.pdf", "success": true, "updated": true, "new_metadata": {"title": "...", "doi": "10.xxxx/...", "doi_provenance": "acatome_bundle"}}
```

CSV format (use `.csv` extension):
```csv
timestamp,pdf_path,success,updated,old_title,new_title,old_doi,new_doi,message,error
```

## Files Changed

- `@/Users/bots/Documents/openclaw-cluster/pips/packages/acatome-extract/src/acatome_extract/pdf_metadata.py` — new module (~500 lines)
- `@/Users/bots/Documents/openclaw-cluster/pips/packages/acatome-extract/src/acatome_extract/cli.py` — added `enrich-pdf` command
- `@/Users/bots/Documents/openclaw-cluster/pips/packages/acatome-extract/pyproject.toml` — added `pdf2doi` optional dependency
- `@/Users/bots/Documents/openclaw-cluster/pips/packages/acatome-extract/tests/test_pdf_metadata.py` — new test file (25 tests)

## CRC / Hash Considerations

Writing metadata to PDFs changes their content hash. This is intentional:
- The `.acatome` bundle remains the canonical source of truth
- The enriched PDF becomes a "finished" archival artifact
- The hash change breaks deduplication by hash alone, but the DOI-based paper ID remains stable
