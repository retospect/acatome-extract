"""Integration tests for sentence-transformers GPU embedding.

These tests require sentence-transformers to be installed:
    uv pip install sentence-transformers

Run with: uv run pytest -m gpu -v
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("sentence_transformers")


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def minilm_model():
    """Load all-MiniLM-L6-v2 once for the module."""
    return st.SentenceTransformer("all-MiniLM-L6-v2", device=_device())


@pytest.mark.gpu
class TestSentenceTransformerDefault:
    def test_encode_dim(self, minilm_model):
        embs = minilm_model.encode(["hello world"], normalize_embeddings=True)
        assert embs.shape == (1, 384)

    def test_encode_batch(self, minilm_model):
        texts = ["quantum computing", "surface codes", "the cat sat on the mat"]
        embs = minilm_model.encode(texts, normalize_embeddings=True)
        assert embs.shape == (3, 384)

    def test_semantic_ranking(self, minilm_model):
        import numpy as np

        texts = [
            "Quantum error correction protects information from noise.",
            "Surface codes have high fault-tolerance thresholds.",
            "The weather today is sunny and warm.",
        ]
        embs = minilm_model.encode(texts, normalize_embeddings=True)
        cos_01 = float(np.dot(embs[0], embs[1]))
        cos_02 = float(np.dot(embs[0], embs[2]))
        assert (
            cos_01 > cos_02
        ), f"Expected quantum↔surface ({cos_01:.3f}) > quantum↔weather ({cos_02:.3f})"

    def test_truncation_mechanics(self, minilm_model):
        """Verify truncation + renormalization produces valid vectors.

        NOTE: all-MiniLM-L6-v2 is NOT a Matryoshka model, so truncation
        may not preserve semantic ranking. We only test mechanics here.
        Matryoshka ranking tests belong with Qwen3-Embedding-4B.
        """
        import numpy as np

        embs = minilm_model.encode(["test sentence"], normalize_embeddings=True)
        trunc = embs[:, :128]
        norms = np.linalg.norm(trunc, axis=1, keepdims=True)
        trunc = trunc / norms

        assert trunc.shape == (1, 128)
        assert abs(float(np.linalg.norm(trunc[0])) - 1.0) < 1e-5


@pytest.mark.gpu
class TestEnrichWithST:
    def test_enrich_default_profile(self, tmp_path):
        """Enrich a bundle using the default Chroma embedder (no ST needed)."""
        import gzip
        import json

        bundle_data = {
            "header": {
                "paper_id": "test:st-embed",
                "slug": "st-embed-test",
                "title": "ST Test",
                "authors": [{"name": "Test"}],
                "year": 2024,
                "doi": None,
                "arxiv_id": None,
                "journal": "",
                "abstract": "Testing.",
                "entry_type": "article",
                "s2_id": None,
                "keywords": [],
                "pdf_hash": "b" * 64,
                "page_count": 1,
                "source": "manual",
                "verified": True,
                "verify_warnings": [],
                "extracted_at": "2024-01-01T00:00:00+00:00",
            },
            "blocks": [
                {
                    "node_id": "test:st-embed-p00-000",
                    "page": 0,
                    "type": "text",
                    "text": "Surface codes are a family of quantum error-correcting codes defined on a 2D lattice.",
                    "section_path": ["1", "Intro"],
                    "bbox": None,
                    "embeddings": {},
                    "summary": None,
                },
            ],
            "enrichment_meta": None,
        }

        bundle_path = tmp_path / "st-test.acatome"
        with gzip.open(bundle_path, "wt", encoding="utf-8") as f:
            json.dump(bundle_data, f)

        from acatome_extract.enrich import enrich

        enrich(bundle_path, profiles=["default"], summarize=False)

        with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
            result = json.load(f)

        block = result["blocks"][0]
        assert "default" in block["embeddings"]
        assert len(block["embeddings"]["default"]) == 384
        assert all(isinstance(x, float) for x in block["embeddings"]["default"])

    def test_get_embedder_st(self):
        """Verify _get_embedder returns a callable for sentence-transformers provider."""
        from acatome_meta.config import EmbedProfile
        from acatome_extract.enrich import _get_embedder

        profile = EmbedProfile(
            model="all-MiniLM-L6-v2",
            dim=384,
            provider="sentence-transformers",
        )
        embedder = _get_embedder(profile)
        assert embedder is not None

        result = embedder(["test sentence"])
        assert len(result) == 1
        assert len(result[0]) == 384
        assert all(isinstance(x, float) for x in result[0])

    def test_get_embedder_st_with_truncation(self):
        """Verify index_dim truncation works."""
        from acatome_meta.config import EmbedProfile
        from acatome_extract.enrich import _get_embedder

        profile = EmbedProfile(
            model="all-MiniLM-L6-v2",
            dim=384,
            provider="sentence-transformers",
            index_dim=128,
        )
        embedder = _get_embedder(profile)
        assert embedder is not None

        result = embedder(["test sentence"])
        assert len(result) == 1
        assert len(result[0]) == 128
