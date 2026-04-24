"""Tests for the migrate CLI command and related helpers."""

from __future__ import annotations

from pathlib import Path

from acatome_extract.bundle import read_bundle, write_bundle
from acatome_extract.cli import _has_llm_summaries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(tmp_path: Path, name: str, blocks: list[dict], em=None) -> Path:
    """Write a minimal bundle with given blocks."""
    data = {
        "header": {
            "paper_id": "sha256:abc123",
            "slug": name,
            "title": "Test",
            "authors": [],
            "year": 2024,
            "doi": None,
            "pdf_hash": "a" * 64,
            "page_count": 1,
            "source": "local",
            "entry_type": "article",
            "verified": True,
            "verify_warnings": [],
            "extracted_at": "2024-01-01T00:00:00+00:00",
        },
        "blocks": blocks,
        "enrichment_meta": em,
    }
    path = tmp_path / f"{name}.acatome"
    write_bundle(data, path)
    return path


# ---------------------------------------------------------------------------
# _has_llm_summaries
# ---------------------------------------------------------------------------


class TestHasLlmSummaries:
    def test_no_summaries(self):
        data = {"blocks": [{"summaries": {}}]}
        assert _has_llm_summaries(data) is False

    def test_rake_only(self):
        data = {"blocks": [{"summaries": {"rake": "keywords"}}]}
        assert _has_llm_summaries(data) is False

    def test_has_llm(self):
        data = {"blocks": [{"summaries": {"llm:ollama/qwen3.5:9b": "summary"}}]}
        assert _has_llm_summaries(data) is True

    def test_old_format_no_summaries_key(self):
        data = {"blocks": [{"summary": "old"}]}
        assert _has_llm_summaries(data) is False

    def test_empty_blocks(self):
        data = {"blocks": []}
        assert _has_llm_summaries(data) is False


# ---------------------------------------------------------------------------
# migrate command
# ---------------------------------------------------------------------------


class TestMigrate:
    def test_migrate_old_format_null_summary(self, tmp_path):
        """Old bundle with summary: None → summaries: {}."""
        bundle = _make_bundle(
            tmp_path,
            "old_null",
            [{"node_id": "a", "type": "text", "text": "Short.", "summary": None}],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(bundle)])
        assert result.exit_code == 0

        data = read_bundle(bundle)
        block = data["blocks"][0]
        assert "summary" not in block
        assert block["summaries"] == {}

    def test_migrate_old_format_with_summary(self, tmp_path):
        """Old bundle with summary string → summaries: {llm:unknown: ...}."""
        bundle = _make_bundle(
            tmp_path,
            "old_with",
            [
                {
                    "node_id": "a",
                    "type": "text",
                    "text": "A" * 100,
                    "summary": "Old LLM summary",
                }
            ],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(bundle)])
        assert result.exit_code == 0

        data = read_bundle(bundle)
        block = data["blocks"][0]
        assert "summary" not in block
        assert block["summaries"]["llm:unknown"] == "Old LLM summary"

    def test_migrate_already_new_format(self, tmp_path):
        """Bundle already in new format → no changes."""
        bundle = _make_bundle(
            tmp_path,
            "already_new",
            [
                {
                    "node_id": "a",
                    "type": "text",
                    "text": "A" * 100,
                    "summaries": {"rake": "keywords here"},
                }
            ],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(bundle)])
        assert result.exit_code == 0
        assert "already current" in result.output

    def test_migrate_adds_rake(self, tmp_path):
        """Migrate adds RAKE summaries to eligible blocks."""
        long_text = (
            "Surface codes provide the highest known error thresholds for "
            "fault-tolerant quantum computing architectures and implementations."
        )
        bundle = _make_bundle(
            tmp_path,
            "needs_rake",
            [
                {
                    "node_id": "a",
                    "type": "text",
                    "text": long_text,
                    "summary": None,
                }
            ],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(bundle)])
        assert result.exit_code == 0

        data = read_bundle(bundle)
        block = data["blocks"][0]
        assert "rake" in block["summaries"]

    def test_migrate_no_rake(self, tmp_path):
        """--no-rake skips RAKE summarization."""
        long_text = (
            "Surface codes provide the highest known error thresholds for "
            "fault-tolerant quantum computing architectures and implementations."
        )
        bundle = _make_bundle(
            tmp_path,
            "no_rake",
            [
                {
                    "node_id": "a",
                    "type": "text",
                    "text": long_text,
                    "summary": None,
                }
            ],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--no-rake", str(bundle)])
        assert result.exit_code == 0

        data = read_bundle(bundle)
        block = data["blocks"][0]
        assert "rake" not in block.get("summaries", {})

    def test_migrate_dry_run(self, tmp_path):
        """--dry-run doesn't modify files."""
        bundle = _make_bundle(
            tmp_path,
            "dryrun",
            [{"node_id": "a", "type": "text", "text": "Short.", "summary": None}],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--dry-run", str(bundle)])
        assert result.exit_code == 0
        assert "[dry-run]" in result.output

        # File should still have old format
        data = read_bundle(bundle)
        assert "summary" in data["blocks"][0]

    def test_migrate_paper_summary(self, tmp_path):
        """Migrates enrichment_meta.paper_summary → paper_summaries."""
        bundle = _make_bundle(
            tmp_path,
            "paper_summ",
            [{"node_id": "a", "type": "text", "text": "X", "summary": None}],
            em={"summarizer": "ollama/qwen3.5:9b", "paper_summary": "Paper overview"},
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(bundle)])
        assert result.exit_code == 0

        data = read_bundle(bundle)
        em = data["enrichment_meta"]
        assert em["paper_summaries"]["llm:unknown"] == "Paper overview"

    def test_migrate_directory(self, tmp_path):
        """Migrate all .acatome files in a directory."""
        _make_bundle(
            tmp_path,
            "one",
            [{"node_id": "a", "type": "text", "text": "X", "summary": None}],
        )
        _make_bundle(
            tmp_path,
            "two",
            [{"node_id": "b", "type": "text", "text": "Y", "summary": "Old"}],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(tmp_path)])
        assert result.exit_code == 0
        assert "2 migrated" in result.output

    def test_migrate_recursive_subdirs(self, tmp_path):
        """Migrate finds .acatome files in subdirectories."""
        subdir = tmp_path / "a"
        subdir.mkdir()
        _make_bundle(
            subdir,
            "deep",
            [{"node_id": "a", "type": "text", "text": "X", "summary": None}],
        )
        from typer.testing import CliRunner

        from acatome_extract.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", str(tmp_path)])
        assert result.exit_code == 0
        assert "1 migrated" in result.output


# ---------------------------------------------------------------------------
# Sidecar .meta.json
# ---------------------------------------------------------------------------


class TestReadSidecar:
    def test_reads_sidecar(self, tmp_path):
        from acatome_extract.pipeline import _read_sidecar

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        meta = tmp_path / "test.meta.json"
        meta.write_text('{"type": "datasheet", "title": "LM317"}')

        result = _read_sidecar(pdf)
        assert result["type"] == "datasheet"
        assert result["title"] == "LM317"

    def test_no_sidecar(self, tmp_path):
        from acatome_extract.pipeline import _read_sidecar

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        assert _read_sidecar(pdf) == {}

    def test_invalid_json(self, tmp_path):
        from acatome_extract.pipeline import _read_sidecar

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake")
        meta = tmp_path / "test.meta.json"
        meta.write_text("not json {{{")
        assert _read_sidecar(pdf) == {}


class TestSidecarOverrideKeys:
    """The sidecar should be able to clear stale s2_id / arxiv_id values
    from the pre-sidecar Crossref/S2 lookup, and should let the user bypass
    the fuzzy-title verification gate when they have manually verified the
    metadata matches the PDF content."""

    def test_override_keys_include_s2_and_arxiv(self):
        from acatome_extract.pipeline import _SIDECAR_OVERRIDE_KEYS

        # Regression: without these keys, the sidecar couldn't clear a stale
        # s2_id, so Store._find_ref kept matching bundles into a wrong ref.
        assert "s2_id" in _SIDECAR_OVERRIDE_KEYS
        assert "arxiv_id" in _SIDECAR_OVERRIDE_KEYS

    def test_explicit_null_clears_field(self):
        """When the sidecar sets a key to ``None`` (JSON null), the field
        is cleared on the header. Empty strings continue to be ignored for
        backward compatibility with older sidecars."""
        from acatome_extract.pipeline import _SIDECAR_OVERRIDE_KEYS

        header = {
            "title": "Original",
            "s2_id": "bogus_id_from_crossref",
            "doi": "10.1/good",
        }
        sidecar = {"s2_id": None, "doi": "10.1/override", "title": ""}

        # Simulate the override loop from extract()
        for key in _SIDECAR_OVERRIDE_KEYS:
            if key not in sidecar:
                continue
            val = sidecar[key]
            if val == "":
                continue
            header[key] = val

        assert header["s2_id"] is None  # explicit null cleared it
        assert header["doi"] == "10.1/override"  # non-empty overrode
        assert header["title"] == "Original"  # empty string ignored


class TestSidecarVerifiedOptIn:
    """``"verified": true`` in the sidecar bypasses the fuzzy-title gate."""

    def test_verified_true_bypasses_verify(self, tmp_path, monkeypatch):
        """Integration-ish test: a sidecar with ``verified: true`` causes the
        pipeline to skip ``verify_metadata`` entirely and mark the bundle
        verified."""
        from acatome_extract import pipeline

        calls = {"verify": 0}

        def fake_verify(header, text):
            calls["verify"] += 1
            return False, ["would have failed"]

        def fake_lookup(path):
            return {
                "title": "Real Title That Doesn't Appear On Page 1",
                "authors": [{"name": "Author"}],
                "doi": "10.1/real",
                "first_pages_text": "unrelated extracted text",
            }

        def fake_pdf_meta(path):
            return {
                "pdf_hash": "abc",
                "page_count": 1,
                "doi": None,
                "first_pages_text": "unrelated",
                "info": {},
            }

        # We only need to check the verify branch, so stub out the expensive
        # parts that follow.
        def fake_blocks(path, paper_id):
            return []

        def fake_write_bundle(bundle, path):
            path.write_bytes(b"stub")
            return path

        monkeypatch.setattr(pipeline, "verify_metadata", fake_verify)
        monkeypatch.setattr(pipeline, "lookup", fake_lookup)
        monkeypatch.setattr(pipeline, "extract_pdf_meta", fake_pdf_meta)
        monkeypatch.setattr(pipeline, "extract_blocks_marker", fake_blocks)
        monkeypatch.setattr(pipeline, "write_bundle", fake_write_bundle)

        pdf = tmp_path / "p.pdf"
        pdf.write_bytes(b"stub")
        (tmp_path / "p.meta.json").write_text(
            '{"title": "Real Title That Doesn\'t Appear On Page 1", "verified": true}'
        )

        pipeline.extract(pdf, output_dir=tmp_path)

        assert calls["verify"] == 0, (
            "verify_metadata must not be called when sidecar sets verified=true"
        )


# ---------------------------------------------------------------------------
# Prompt provenance constants
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_block_prompt_template_exists(self):
        from acatome_extract.enrich import _BLOCK_PROMPT_TEMPLATE

        assert "telegram" in _BLOCK_PROMPT_TEMPLATE.lower()
        assert "semicolon" in _BLOCK_PROMPT_TEMPLATE.lower()

    def test_paper_prompt_template_exists(self):
        from acatome_extract.enrich import _PAPER_PROMPT_TEMPLATE

        assert "telegram" in _PAPER_PROMPT_TEMPLATE.lower()
        assert "line 1" in _PAPER_PROMPT_TEMPLATE.lower()


# ---------------------------------------------------------------------------
# Datasheet / doc_type support
# ---------------------------------------------------------------------------


class TestLocalHeader:
    def test_local_header_from_pdf_meta(self):
        from acatome_extract.pipeline import _local_header

        pdf_path = Path("/tmp/TI_LM317_Datasheet.pdf")
        pdf_meta = {
            "info": {
                "title": "LM317 3-Terminal Adjustable Regulator",
                "author": "Texas Instruments",
                "creationDate": "D:20220415120000",
            },
            "doi": None,
            "pdf_hash": "b" * 64,
            "page_count": 24,
        }
        header = _local_header(pdf_path, pdf_meta, "datasheet")
        assert header["title"] == "LM317 3-Terminal Adjustable Regulator"
        assert header["authors"] == [{"name": "Texas Instruments"}]
        assert header["year"] == 2022
        assert header["entry_type"] == "datasheet"
        assert header["source"] == "local"

    def test_local_header_no_title_uses_filename(self):
        from acatome_extract.pipeline import _local_header

        pdf_path = Path("/tmp/TI_LM317_Datasheet.pdf")
        pdf_meta = {
            "info": {},
            "doi": None,
            "pdf_hash": "c" * 64,
            "page_count": 10,
        }
        header = _local_header(pdf_path, pdf_meta, "manual")
        assert header["title"] == "TI LM317 Datasheet"
        assert header["authors"] == []
        assert header["year"] is None
        assert header["entry_type"] == "manual"

    def test_local_doc_types_set(self):
        from acatome_extract.pipeline import _LOCAL_DOC_TYPES

        assert "datasheet" in _LOCAL_DOC_TYPES
        assert "manual" in _LOCAL_DOC_TYPES
        assert "techreport" in _LOCAL_DOC_TYPES
        assert "notes" in _LOCAL_DOC_TYPES
        assert "other" in _LOCAL_DOC_TYPES
        assert "article" not in _LOCAL_DOC_TYPES
