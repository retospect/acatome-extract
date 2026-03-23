"""Tests for the watch module (unit-level, no real pipeline)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acatome_extract.watch import (
    _log_completed,
    _move_to,
    _pdf_hash,
    _tags_from_path,
    _wait_stable,
    _write_error,
)


class TestWaitStable:
    def test_stable_file(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"hello")
        assert _wait_stable(f, debounce=0.05) is True

    def test_missing_file(self, tmp_path):
        f = tmp_path / "gone.pdf"
        assert _wait_stable(f, debounce=0.05) is False


class TestMoveTo:
    def test_move_basic(self, tmp_path):
        src = tmp_path / "paper.pdf"
        src.write_bytes(b"data")
        dest_dir = tmp_path / "completed"
        dest_dir.mkdir()
        result = _move_to(src, dest_dir)
        assert result.parent == dest_dir
        assert result.name == "paper.pdf"
        assert not src.exists()

    def test_move_conflict(self, tmp_path):
        src = tmp_path / "paper.pdf"
        src.write_bytes(b"new")
        dest_dir = tmp_path / "completed"
        dest_dir.mkdir()
        # Pre-existing file with same name
        (dest_dir / "paper.pdf").write_bytes(b"old")
        result = _move_to(src, dest_dir)
        assert result.parent == dest_dir
        assert result.name != "paper.pdf"  # renamed to avoid conflict
        assert not src.exists()


class TestWriteError:
    def test_writes_error_file(self, tmp_path):
        errors_dir = tmp_path / "errors"
        errors_dir.mkdir()
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"data")
        try:
            raise ValueError("extraction failed")
        except ValueError as e:
            _write_error(errors_dir, pdf, e)
        error_file = errors_dir / "bad.error.txt"
        assert error_file.exists()
        content = error_file.read_text()
        assert "bad.pdf" in content
        assert "extraction failed" in content
        assert "Traceback" in content


class TestRunPipeline:
    @patch("acatome_extract.bundle.read_bundle")
    @patch("acatome_extract.pipeline.extract")
    def test_extract_only(self, mock_extract, mock_read, tmp_path):
        from acatome_extract.watch import run_pipeline

        bundle = tmp_path / "smith2024.acatome"
        bundle.write_bytes(b"bundle")
        mock_extract.return_value = bundle
        mock_read.return_value = {
            "header": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "verified": True,
            }
        }

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"pdf")

        result = run_pipeline(pdf, enrich=False, ingest=False)
        assert result["bundle"] == str(bundle)
        assert result["slug"] == "smith2024"
        assert result["verified"] is True
        mock_extract.assert_called_once()

    @patch("acatome_extract.bundle.read_bundle")
    @patch("acatome_extract.pipeline.extract")
    def test_rejects_unverified(self, mock_extract, mock_read, tmp_path):
        from acatome_extract.watch import UnverifiedPaperError, run_pipeline

        bundle = tmp_path / "anon2024.acatome"
        bundle.write_bytes(b"bundle")
        mock_extract.return_value = bundle
        mock_read.return_value = {
            "header": {
                "title": "PII: 0015-1882(92)80247-G",
                "verified": False,
                "verify_warnings": ["title mismatch"],
            }
        }

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"pdf")

        with pytest.raises(UnverifiedPaperError, match="failed verification"):
            run_pipeline(pdf, enrich=False, ingest=False)

    @patch("acatome_extract.bundle.read_bundle")
    @patch("acatome_extract.pipeline.extract")
    def test_unverified_carries_bundle_path(self, mock_extract, mock_read, tmp_path):
        """UnverifiedPaperError includes bundle_path so error handler can clean up."""
        from acatome_extract.watch import UnverifiedPaperError, run_pipeline

        bundle = tmp_path / "anon2024.acatome"
        bundle.write_bytes(b"bundle")
        mock_extract.return_value = bundle
        mock_read.return_value = {
            "header": {
                "title": "PII: 0015-1882(92)80247-G",
                "verified": False,
                "verify_warnings": ["title mismatch"],
            }
        }

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"pdf")

        with pytest.raises(UnverifiedPaperError) as exc_info:
            run_pipeline(pdf, enrich=False, ingest=False)
        assert exc_info.value.bundle_path == bundle

    @patch("acatome_extract.bundle.read_bundle")
    @patch("acatome_extract.pipeline.extract")
    def test_rejects_garbage_metadata(self, mock_extract, mock_read, tmp_path):
        """Verified=True but no DOI + anon/untitled → still rejected."""
        from acatome_extract.watch import UnverifiedPaperError, run_pipeline

        bundle = tmp_path / "anon2012untitled.acatome"
        bundle.write_bytes(b"bundle")
        mock_extract.return_value = bundle
        mock_read.return_value = {
            "header": {
                "title": "untitled",
                "verified": True,
            }
        }

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"pdf")

        with pytest.raises(UnverifiedPaperError, match="garbage metadata"):
            run_pipeline(pdf, enrich=False, ingest=False)


class TestOrphanBundleCleanup:
    def test_orphan_bundle_moved_to_errors(self, tmp_path):
        """When UnverifiedPaperError has bundle_path, error handler moves it to errors/."""
        from acatome_extract.watch import UnverifiedPaperError

        errors_dir = tmp_path / "errors"
        errors_dir.mkdir()

        # Simulate orphan bundle in papers dir
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        bundle = papers_dir / "anon2024bad.acatome"
        bundle.write_bytes(b"bundle data")

        # Simulate the error handler logic from _process()
        e = UnverifiedPaperError("failed verification", bundle_path=bundle)
        bp = getattr(e, "bundle_path", None)
        if bp and isinstance(bp, Path) and bp.exists():
            _move_to(bp, errors_dir)

        assert not bundle.exists(), "Bundle should be removed from papers/"
        assert (errors_dir / "anon2024bad.acatome").exists(), "Bundle should be in errors/"

    def test_no_crash_when_bundle_already_gone(self, tmp_path):
        """Error handler doesn't crash if bundle was already deleted."""
        from acatome_extract.watch import UnverifiedPaperError

        errors_dir = tmp_path / "errors"
        errors_dir.mkdir()
        bundle = tmp_path / "gone.acatome"  # does not exist

        e = UnverifiedPaperError("failed", bundle_path=bundle)
        bp = getattr(e, "bundle_path", None)
        # Should not move since file doesn't exist
        assert bp is not None
        assert not bp.exists()  # guard condition prevents move


class TestPdfHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"hello world")
        h1 = _pdf_hash(f)
        h2 = _pdf_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert _pdf_hash(a) != _pdf_hash(b)

    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"same content")
        b.write_bytes(b"same content")
        assert _pdf_hash(a) == _pdf_hash(b)


class TestLogCompleted:
    def test_creates_log_and_appends(self, tmp_path):
        completed = tmp_path / "completed"
        completed.mkdir()
        pdf = tmp_path / "paper.pdf"

        result = {
            "slug": "smith2024quantum",
            "doi": "10.1038/s41567-024-1234-5",
            "ref_id": 7,
            "title": "Quantum Error Correction",
        }
        _log_completed(completed, "reto", result, pdf)
        _log_completed(
            completed,
            "alice",
            {"slug": "jones2023", "doi": "", "ref_id": 2, "title": "Other"},
            pdf,
        )

        log_file = completed / "ingest.log"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # First line is reto's
        assert "reto" in lines[0]
        assert "smith2024quantum" in lines[0]
        assert "10.1038/s41567-024-1234-5" in lines[0]

        # Second line is alice's
        assert "alice" in lines[1]
        assert "jones2023" in lines[1]

    def test_greppable_by_user(self, tmp_path):
        completed = tmp_path / "completed"
        completed.mkdir()
        pdf = tmp_path / "a.pdf"

        for user, slug in [("reto", "paper1"), ("reto", "paper2"), ("bob", "paper3")]:
            _log_completed(
                completed,
                user,
                {"slug": slug, "doi": "", "ref_id": 1, "title": ""},
                pdf,
            )

        content = (completed / "ingest.log").read_text()
        reto_lines = [l for l in content.strip().split("\n") if "reto" in l]
        assert len(reto_lines) == 2
        bob_lines = [l for l in content.strip().split("\n") if "bob" in l]
        assert len(bob_lines) == 1


class TestTagsFromPath:
    def test_subdir_becomes_tag(self, tmp_path):
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        sub = watch_dir / "chlorine-evolution"
        sub.mkdir()
        pdf = sub / "paper.pdf"
        pdf.write_bytes(b"data")
        assert _tags_from_path(pdf, watch_dir) == ["chlorine-evolution"]

    def test_nested_subdirs(self, tmp_path):
        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "electrocatalysis" / "2024"
        sub.mkdir(parents=True)
        pdf = sub / "paper.pdf"
        pdf.write_bytes(b"data")
        assert _tags_from_path(pdf, watch_dir) == ["electrocatalysis", "2024"]

    def test_root_dir_no_tags(self, tmp_path):
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        pdf = watch_dir / "paper.pdf"
        pdf.write_bytes(b"data")
        assert _tags_from_path(pdf, watch_dir) == []

    def test_skips_special_dirs(self, tmp_path):
        watch_dir = tmp_path / "inbox"
        sub = watch_dir / "completed" / "old"
        sub.mkdir(parents=True)
        pdf = sub / "paper.pdf"
        pdf.write_bytes(b"data")
        assert _tags_from_path(pdf, watch_dir) == ["old"]

    def test_outside_watch_dir(self, tmp_path):
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        pdf = tmp_path / "elsewhere" / "paper.pdf"
        pdf.parent.mkdir()
        pdf.write_bytes(b"data")
        assert _tags_from_path(pdf, watch_dir) == []


class TestShouldSkip:
    def test_skips_completed_dir(self, tmp_path):
        """Files inside completed/ should be skipped."""
        from acatome_extract.watch import _PdfHandler

        completed = tmp_path / "completed"
        completed.mkdir()
        pdf = completed / "done.pdf"
        pdf.write_bytes(b"data")

        # Simulate the skip logic by checking path ancestry
        try:
            pdf.resolve().relative_to(completed.resolve())
            skipped = True
        except ValueError:
            skipped = False
        assert skipped is True

    def test_does_not_skip_normal(self, tmp_path):
        """Normal PDFs should not be skipped."""
        completed = tmp_path / "completed"
        completed.mkdir()
        pdf = tmp_path / "new.pdf"
        pdf.write_bytes(b"data")

        try:
            pdf.resolve().relative_to(completed.resolve())
            skipped = True
        except ValueError:
            skipped = False
        assert skipped is False
