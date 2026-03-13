# acatome-extract

PDF extraction and enrichment pipeline for scientific papers. Converts PDFs into structured, searchable bundles with block-level summaries and embeddings.

## Features

- **Marker PDF extraction** — structured block extraction with headings, tables, figures
- **Fitz fallback** — recursive character chunking when Marker is unavailable
- **LLM enrichment** — block and paper summaries via Ollama or litellm
- **Embeddings** — sentence-transformer embeddings for semantic search
- **File watcher** — `acatome-extract watch` monitors an inbox folder
- **Bundle format** — `.acatome` companion files for sharing pre-built extractions
- **CLI** — `acatome-extract` command for extract, enrich, and watch workflows

## Installation

```bash
uv pip install -e .
```

On **macOS/Linux** this includes Marker for structured PDF extraction.
On **Windows** it installs with the lighter pymupdf (fitz) backend by default.
To add Marker on Windows (requires C build tools):

```bash
uv pip install -e ".[marker]"
```

With GPU acceleration (embeddings + torch):

```bash
uv pip install -e ".[gpu]"
```

Everything at once:

```bash
uv pip install -e ".[full]"
```

## Usage

```python
from acatome_extract.pipeline import extract

bundle = extract("/path/to/paper.pdf")
```

## CLI

```bash
# Extract (RAKE summaries included automatically, no LLM needed)
acatome-extract extract paper.pdf
acatome-extract extract --type datasheet TI_LM317.pdf   # non-article types

# Enrich — embeddings only by default; add --summarize for LLM summaries
acatome-extract enrich /path/to/bundle
acatome-extract enrich --summarize /path/to/bundle       # enable LLM summaries
acatome-extract enrich --summarize --skip-existing dir/   # incremental LLM pass

# Watch — extract + embed + ingest; LLM summaries off by default
acatome-extract watch ~/papers/inbox
acatome-extract watch ~/papers/inbox --summarize          # enable LLM summaries

# Migrate old bundles to new summaries dict format + add RAKE
acatome-extract migrate ~/.acatome/papers
acatome-extract migrate ~/.acatome/papers --dry-run       # preview changes

# Supplements
acatome-extract attach parent-slug supplement.pdf --name s1
```

### Summaries

Extraction always generates **RAKE** (extractive keyword) summaries — instant, no LLM required. LLM-based summaries are opt-in via `--summarize` and require an Ollama or litellm-compatible model.

RAKE summaries are used as the default for search and display. To add LLM summaries later:

```bash
acatome-extract enrich --summarize --skip-existing ~/.acatome/papers
```

### Sidecar metadata

Place a `<stem>.meta.json` alongside any PDF to override metadata:

```json
{"type": "datasheet", "title": "LM317 Regulator", "author": "Texas Instruments", "year": 2022}
```

Supported fields: `type`, `title`, `author` (string or list), `year`, `doi`, `abstract`, `journal`.

## Dependencies

- **acatome-meta** — metadata lookup and verification
- **marker-pdf** — structured PDF extraction
- **litellm** / **Ollama** — LLM-based enrichment

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
