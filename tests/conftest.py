"""Shared fixtures for acatome-extract tests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_bundle(tmp_path) -> Path:
    """Create a minimal .acatome bundle for testing."""
    bundle = {
        "header": {
            "paper_id": "doi:10.1038/s41567-024-1234-5",
            "slug": "smith2024quantum",
            "title": "Quantum Error Correction in Practice",
            "authors": [{"name": "Smith, John"}],
            "year": 2024,
            "doi": "10.1038/s41567-024-1234-5",
            "arxiv_id": None,
            "journal": "Nature Physics",
            "abstract": "We present...",
            "entry_type": "article",
            "s2_id": None,
            "keywords": [],
            "pdf_hash": "a" * 64,
            "page_count": 12,
            "source": "crossref",
            "verified": True,
            "verify_warnings": [],
            "extracted_at": "2024-01-15T12:00:00+00:00",
        },
        "blocks": [
            {
                "node_id": "doi:10.1038/s41567-024-1234-5-p00-000",
                "page": 0,
                "type": "text",
                "text": "Quantum error correction is essential...",
                "section_path": ["1", "Introduction"],
                "bbox": [72, 100, 540, 200],
                "embeddings": {},
                "summary": None,
            }
        ],
        "enrichment_meta": None,
    }
    path = tmp_path / "smith2024quantum.acatome"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(bundle, f)
    return path
