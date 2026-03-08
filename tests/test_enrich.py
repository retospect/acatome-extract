"""Tests for the enrich module."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acatome_extract.bundle import read_bundle
from acatome_extract.enrich import (
    _embed_blocks,
    _summarize_blocks,
    enrich,
)


@pytest.fixture
def enrichable_bundle(tmp_path) -> Path:
    """Bundle with blocks ready for enrichment."""
    data = {
        "header": {
            "paper_id": "doi:10.1038/test",
            "slug": "test2024enrich",
            "title": "Test Paper",
            "authors": [{"name": "Doe"}],
            "year": 2024,
            "doi": "10.1038/test",
            "arxiv_id": None,
            "journal": "Nature",
            "abstract": "Test abstract.",
            "entry_type": "article",
            "s2_id": None,
            "keywords": [],
            "pdf_hash": "e" * 64,
            "page_count": 2,
            "source": "crossref",
            "verified": True,
            "verify_warnings": [],
            "extracted_at": "2024-01-01T00:00:00+00:00",
        },
        "blocks": [
            {
                "node_id": "doi:10.1038/test-p00-000",
                "page": 0,
                "type": "text",
                "text": "Quantum error correction is fundamental to building scalable quantum computers with low logical error rates.",
                "section_path": ["1", "Introduction"],
                "bbox": None,
                "embeddings": {},
                "summary": None,
            },
            {
                "node_id": "doi:10.1038/test-p00-001",
                "page": 0,
                "type": "text",
                "text": "Surface codes provide the highest known thresholds for fault-tolerant quantum computing architectures.",
                "section_path": ["2", "Surface Codes"],
                "bbox": None,
                "embeddings": {},
                "summary": None,
            },
            {
                "node_id": "doi:10.1038/test-p00-002",
                "page": 0,
                "type": "section_header",
                "text": "Results",
                "section_path": ["3"],
                "bbox": None,
                "embeddings": {},
                "summary": None,
            },
            {
                "node_id": "doi:10.1038/test-p01-000",
                "page": 1,
                "type": "text",
                "text": "Short.",
                "section_path": ["3", "Results"],
                "bbox": None,
                "embeddings": {},
                "summary": None,
            },
        ],
        "enrichment_meta": None,
    }
    path = tmp_path / "test2024enrich.acatome"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path


class TestEmbedBlocks:
    def test_embed_with_mock(self):
        blocks = [
            {"node_id": "a", "type": "text", "text": "Hello world", "embeddings": {}},
            {
                "node_id": "b",
                "type": "section_header",
                "text": "Title",
                "embeddings": {},
            },
            {"node_id": "c", "type": "text", "text": "Goodbye", "embeddings": {}},
        ]
        embedder = lambda texts: [[0.1] * 384 for _ in texts]

        result = _embed_blocks(blocks, embedder, "default")
        assert "default" in result[0]["embeddings"]
        assert len(result[0]["embeddings"]["default"]) == 384
        assert "default" not in result[1]["embeddings"]  # section_header skipped
        assert "default" in result[2]["embeddings"]

    def test_embed_skip_empty(self):
        blocks = [
            {"node_id": "a", "type": "text", "text": "", "embeddings": {}},
        ]
        embedder = MagicMock(return_value=[])
        result = _embed_blocks(blocks, embedder, "default")
        embedder.assert_not_called()

    def test_embed_handles_error(self):
        blocks = [
            {"node_id": "a", "type": "text", "text": "Hello", "embeddings": {}},
        ]
        embedder = MagicMock(side_effect=RuntimeError("model unavailable"))
        result = _embed_blocks(blocks, embedder, "default")
        assert "default" not in result[0]["embeddings"]


class TestSummarizeBlocks:
    def test_summarize_with_mock_llm(self):
        blocks = [
            {"node_id": "a", "type": "text", "text": "A" * 100, "summary": None},
            {
                "node_id": "b",
                "type": "section_header",
                "text": "Title",
                "summary": None,
            },
            {"node_id": "c", "type": "text", "text": "Short.", "summary": None},
        ]

        with patch("acatome_extract.enrich._get_llm") as mock_get:
            mock_llm = MagicMock(return_value="Test summary.")
            mock_get.return_value = mock_llm

            result = _summarize_blocks(blocks, "ollama:test")
            assert result[0]["summary"] == "Test summary."
            assert result[1]["summary"] is None  # section_header
            assert result[2]["summary"] is None  # too short

    def test_summarize_no_llm(self):
        blocks = [{"node_id": "a", "type": "text", "text": "A" * 100, "summary": None}]
        result = _summarize_blocks(blocks, "")
        assert result[0]["summary"] is None


class TestEnrichE2E:
    def test_enrich_embeddings_only(self, enrichable_bundle):
        """Enrich with Chroma default embeddings, no summarization."""
        result_path = enrich(
            enrichable_bundle,
            profiles=["default"],
            summarize=False,
        )
        data = read_bundle(result_path)

        # Text blocks should have embeddings
        text_blocks = [b for b in data["blocks"] if b["type"] == "text"]
        embedded = [b for b in text_blocks if b["embeddings"].get("default")]
        assert len(embedded) >= 1  # At least the longer blocks

        # Section header should NOT have embeddings
        headers = [b for b in data["blocks"] if b["type"] == "section_header"]
        for h in headers:
            assert "default" not in h.get("embeddings", {})

    def test_enrich_records_meta(self, enrichable_bundle):
        enrich(enrichable_bundle, profiles=["default"], summarize=False)
        data = read_bundle(enrichable_bundle)
        assert data["enrichment_meta"] is not None
        assert "default" in data["enrichment_meta"]["profiles"]

    def test_strip_after_enrich(self, enrichable_bundle):
        enrich(enrichable_bundle, profiles=["default"], summarize=False)
        data = read_bundle(enrichable_bundle)

        # Strip embeddings
        for block in data["blocks"]:
            embs = block.get("embeddings", {})
            if "default" in embs:
                del embs["default"]

        from acatome_extract.bundle import write_bundle

        write_bundle(data, enrichable_bundle)
        data2 = read_bundle(enrichable_bundle)
        for b in data2["blocks"]:
            assert "default" not in b.get("embeddings", {})
