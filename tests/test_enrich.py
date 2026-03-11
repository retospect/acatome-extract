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
    _is_non_latin,
    _summarize_blocks,
    _translate_slug,
    enrich,
)
from precis_summary import pick_best_summary


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
                "summaries": {},
            },
            {
                "node_id": "doi:10.1038/test-p00-001",
                "page": 0,
                "type": "text",
                "text": "Surface codes provide the highest known thresholds for fault-tolerant quantum computing architectures.",
                "section_path": ["2", "Surface Codes"],
                "bbox": None,
                "embeddings": {},
                "summaries": {},
            },
            {
                "node_id": "doi:10.1038/test-p00-002",
                "page": 0,
                "type": "section_header",
                "text": "Results",
                "section_path": ["3"],
                "bbox": None,
                "embeddings": {},
                "summaries": {},
            },
            {
                "node_id": "doi:10.1038/test-p01-000",
                "page": 1,
                "type": "text",
                "text": "Short.",
                "section_path": ["3", "Results"],
                "bbox": None,
                "embeddings": {},
                "summaries": {},
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
            {"node_id": "a", "type": "text", "text": "A" * 100, "summaries": {}},
            {
                "node_id": "b",
                "type": "section_header",
                "text": "Title",
                "summaries": {},
            },
            {"node_id": "c", "type": "text", "text": "Short.", "summaries": {}},
        ]

        with patch("acatome_extract.enrich._get_llm") as mock_get:
            mock_llm = MagicMock(return_value="Test summary.")
            mock_get.return_value = mock_llm

            result = _summarize_blocks(blocks, "ollama:test", summary_key="llm:ollama:test")
            assert result[0]["summaries"]["llm:ollama:test"] == "Test summary."
            assert result[1]["summaries"] == {}  # section_header
            assert result[2]["summaries"] == {}  # too short

    def test_summarize_no_llm(self):
        blocks = [{"node_id": "a", "type": "text", "text": "A" * 100, "summaries": {}}]
        result = _summarize_blocks(blocks, "")
        assert result[0]["summaries"] == {}


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


class TestIsNonLatin:
    def test_english(self):
        assert _is_non_latin("Quantum Error Correction") is False

    def test_chinese(self):
        assert _is_non_latin("零间隙CO₂电解实验研究") is True

    def test_japanese(self):
        assert _is_non_latin("量子コンピュータの研究") is True

    def test_mixed_mostly_english(self):
        assert _is_non_latin("A New 催化 Catalyst Design") is False

    def test_mixed_mostly_chinese(self):
        assert _is_non_latin("新型催化剂 CO2 设计与应用研究") is True

    def test_accented_latin(self):
        assert _is_non_latin("Über die Wärmeleitfähigkeit") is False

    def test_empty(self):
        assert _is_non_latin("") is False

    def test_numbers_only(self):
        assert _is_non_latin("12345") is False


class TestTranslateSlug:
    def test_latin_title_unchanged(self, tmp_path):
        data = {
            "header": {
                "slug": "smith2024quantum",
                "title": "Quantum Error Correction",
                "authors": [{"name": "Smith"}],
                "year": 2024,
            }
        }
        bundle_path = tmp_path / "smith2024quantum.acatome"
        bundle_path.touch()
        result = _translate_slug(data, bundle_path, "ollama/test")
        assert result == bundle_path
        assert data["header"]["slug"] == "smith2024quantum"

    def test_chinese_title_translated(self, tmp_path):
        data = {
            "header": {
                "slug": "zhang2023a1b2c3",
                "title": "零间隙CO₂电解实验研究",
                "authors": [{"name": "Zhang, Wei"}],
                "year": 2023,
            }
        }
        bundle_path = tmp_path / "zhang2023a1b2c3.acatome"
        bundle_path.touch()

        with patch("acatome_extract.enrich._get_llm") as mock_get:
            mock_llm = MagicMock(return_value="electrolysis")
            mock_get.return_value = mock_llm
            result = _translate_slug(data, bundle_path, "ollama/test")

        assert data["header"]["slug"] == "zhang2023electrolysis"
        assert result.name == "zhang2023electrolysis.acatome"
        assert result.exists()
        assert not bundle_path.exists()  # old file renamed

    def test_renames_companion_pdf(self, tmp_path):
        data = {
            "header": {
                "slug": "li2024abcdef",
                "title": "新型催化剂的设计与合成",
                "authors": [{"name": "Li"}],
                "year": 2024,
            }
        }
        bundle_path = tmp_path / "li2024abcdef.acatome"
        bundle_path.touch()
        pdf_path = tmp_path / "li2024abcdef.pdf"
        pdf_path.touch()

        with patch("acatome_extract.enrich._get_llm") as mock_get:
            mock_llm = MagicMock(return_value="catalyst")
            mock_get.return_value = mock_llm
            result = _translate_slug(data, bundle_path, "ollama/test")

        assert (tmp_path / "li2024catalyst.pdf").exists()
        assert not pdf_path.exists()

    def test_no_llm_unchanged(self, tmp_path):
        data = {
            "header": {
                "slug": "anon2023abcdef",
                "title": "日本語タイトル",
                "authors": [],
                "year": 2023,
            }
        }
        bundle_path = tmp_path / "anon2023abcdef.acatome"
        bundle_path.touch()

        with patch("acatome_extract.enrich._get_llm", return_value=None):
            result = _translate_slug(data, bundle_path, "")

        assert result == bundle_path
        assert data["header"]["slug"] == "anon2023abcdef"
