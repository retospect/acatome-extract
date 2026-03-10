"""Tests for deterministic ID generation."""

from __future__ import annotations

import pytest

from acatome_extract.ids import make_node_id, make_paper_id, make_slug


class TestMakePaperId:
    def test_arxiv_priority(self):
        assert (
            make_paper_id(doi="10.1038/test", arxiv_id="2401.12345")
            == "arxiv:2401.12345"
        )

    def test_doi_when_no_arxiv(self):
        assert make_paper_id(doi="10.1038/test") == "doi:10.1038/test"

    def test_sha256_fallback(self):
        result = make_paper_id(pdf_hash="abcdef123456789012345678")
        assert result == "sha256:abcdef123456"

    def test_no_ids_raises(self):
        with pytest.raises(ValueError):
            make_paper_id()

    def test_deterministic(self):
        a = make_paper_id(doi="10.1038/test")
        b = make_paper_id(doi="10.1038/test")
        assert a == b


class TestMakeSlug:
    def test_basic(self):
        assert (
            make_slug([{"name": "Smith, John"}], 2024, "Quantum Error Correction")
            == "smith2024quantum"
        )

    def test_skip_stopwords(self):
        assert (
            make_slug([{"name": "Jones"}], 2023, "A New Approach to Surface Codes")
            == "jones2023approach"
        )

    def test_no_author(self):
        assert make_slug([], 2020, "Thermal Decomposition") == "anon2020thermal"

    def test_no_year(self):
        assert make_slug([{"name": "Doe, Jane"}], None, "Some Title") == "doe0000some"

    def test_accented_name(self):
        slug = make_slug([{"name": "Müller, Hans"}], 2021, "Chicken Little")
        assert slug == "muller2021chicken"

    def test_empty_title(self):
        assert make_slug([{"name": "Smith"}], 2024, "") == "smith2024untitled"

    def test_chinese_title(self):
        slug = make_slug([{"name": "张, 伟"}], 2023, "零间隙CO₂电解实验研究")
        # Author name is non-Latin → "anon", title has no ASCII words → short hash
        assert slug.startswith("anon2023")
        assert len(slug) > len("anon2023")  # has a keyword hash

    def test_mixed_cjk_english_title(self):
        slug = make_slug([{"name": "Li, Wei"}], 2024, "新型 Catalyst Design for CO2 Reduction")
        # "catalyst" is first ASCII content word (skips stopword "for")
        assert slug == "li2024catalyst"

    def test_non_latin_author(self):
        slug = make_slug([{"name": "田中太郎"}], 2022, "Thermal Analysis")
        # Japanese name → no ASCII → "anon"
        assert slug == "anon2022thermal"

    def test_chinese_title_deterministic(self):
        a = make_slug([], 2023, "零间隙CO₂电解实验研究")
        b = make_slug([], 2023, "零间隙CO₂电解实验研究")
        assert a == b


class TestMakeNodeId:
    def test_format(self):
        result = make_node_id("doi:10.1038/test", 4, 20)
        assert result == "doi:10.1038/test-p04-020"

    def test_zero_padded(self):
        result = make_node_id("arxiv:2401.12345", 0, 0)
        assert result == "arxiv:2401.12345-p00-000"

    def test_deterministic(self):
        a = make_node_id("doi:10.1038/test", 1, 5)
        b = make_node_id("doi:10.1038/test", 1, 5)
        assert a == b
