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

With GPU acceleration:

```bash
uv pip install -e ".[gpu]"
```

## Usage

```python
from acatome_extract.pipeline import extract

bundle = extract("/path/to/paper.pdf")
```

## CLI

```bash
acatome-extract extract paper.pdf
acatome-extract enrich /path/to/bundle
acatome-extract watch ~/papers/inbox
acatome-extract attach parent-slug supplement.pdf --name s1
```

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
