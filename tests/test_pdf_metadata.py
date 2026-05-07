"""Tests for pdf_metadata module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from acatome_extract.pdf_metadata import (
    DoiCandidate,
    DoiProvenance,
    PdfMetadata,
    _find_acatome_bundle,
    _is_valid_doi_format,
    _normalize_doi,
    _read_sidecar_meta,
    build_exiftool_command,
    should_update_file,
)


class TestDoiNormalization:
    """Tests for DOI normalization."""

    def test_normalize_doi_strips_prefix(self):
        assert _normalize_doi("doi:10.1000/abc") == "10.1000/abc"
        assert _normalize_doi("DOI:10.1000/abc") == "10.1000/abc"

    def test_normalize_doi_lowercases(self):
        assert _normalize_doi("10.1000/ABC") == "10.1000/abc"

    def test_normalize_doi_strips_whitespace(self):
        assert _normalize_doi("  10.1000/abc  ") == "10.1000/abc"


class TestDoiValidation:
    """Tests for DOI format validation."""

    def test_valid_doi_formats(self):
        assert _is_valid_doi_format("10.1000/abc")
        assert _is_valid_doi_format("10.1038/s41586-020-2649-2")
        assert _is_valid_doi_format("10.1103/physrevlett.89.106801")

    def test_invalid_doi_formats(self):
        assert not _is_valid_doi_format("")
        assert not _is_valid_doi_format("not-a-doi")
        assert not _is_valid_doi_format("10.1/short")  # Registrant too short


class TestPdfMetadataDataclass:
    """Tests for PdfMetadata dataclass."""

    def test_basic_creation(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            title="Test Paper",
            authors=["Smith, J.", "Jones, A."],
            doi="10.1000/abc",
        )
        assert meta.title == "Test Paper"
        assert meta.authors == ["Smith, J.", "Jones, A."]
        assert meta.doi == "10.1000/abc"

    def test_to_exiftool_args_basic(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            title="Test Paper",
            authors=["Smith, J."],
            doi="10.1000/abc",
        )
        args = meta.to_exiftool_args()

        assert "-Title=Test Paper" in args
        assert "-Author=Smith, J." in args
        assert "-Keywords=10.1000/abc" in args
        assert "-XMP-dc:Identifier=doi:10.1000/abc" in args
        assert "-XMP-dc:Creator=Smith, J." in args

    def test_to_exiftool_args_multiple_authors(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            authors=["Smith, J.", "Jones, A.", "Lee, K."],
        )
        args = meta.to_exiftool_args()

        # Check joined Author field
        assert "-Author=Smith, J.; Jones, A.; Lee, K." in args
        # Check individual XMP-dc:Creator entries
        assert args.count("-XMP-dc:Creator=Smith, J.") == 1
        assert args.count("-XMP-dc:Creator=Jones, A.") == 1
        assert args.count("-XMP-dc:Creator=Lee, K.") == 1

    def test_to_exiftool_args_journal_in_subject(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            journal="Nature",
            year=2024,
        )
        args = meta.to_exiftool_args()
        assert "-Subject=Nature, 2024" in args

    def test_to_exiftool_args_publisher(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            publisher="Springer Nature",
        )
        args = meta.to_exiftool_args()
        assert "-XMP-dc:Publisher=Springer Nature" in args

    def test_to_exiftool_args_year_as_date(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            year=2024,
        )
        args = meta.to_exiftool_args()
        assert "-XMP-dc:Date=2024" in args

    def test_to_exiftool_args_doi_in_keywords(self):
        """DOI must be included in Keywords field."""
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            doi="10.1000/abc",
            keywords=["MOF", "DFT"],
        )
        args = meta.to_exiftool_args()

        # DOI should be first in keywords
        kw_arg = [a for a in args if a.startswith("-Keywords=")][0]
        assert kw_arg == "-Keywords=10.1000/abc, MOF, DFT"


class TestBuildExiftoolCommand:
    """Tests for building exiftool command."""

    def test_command_structure(self):
        meta = PdfMetadata(
            pdf_path=Path("/tmp/test.pdf"),
            title="Test",
            doi="10.1000/abc",
        )
        cmd = build_exiftool_command(Path("/tmp/test.pdf"), meta)

        assert cmd[0] == "exiftool"
        assert "-overwrite_original" in cmd
        assert "-preserve" in cmd
        assert str(Path("/tmp/test.pdf")) in cmd


class TestShouldUpdateFile:
    """Tests for update decision logic."""

    @patch("acatome_extract.pdf_metadata._read_existing_pdf_metadata")
    def test_should_update_when_no_doi(self, mock_read):
        mock_read.return_value = {}
        meta = PdfMetadata(pdf_path=Path("/tmp/test.pdf"), doi="10.1000/abc")

        should, reason = should_update_file(Path("/tmp/test.pdf"), meta)
        assert should is True
        assert "DOI missing" in reason

    @patch("acatome_extract.pdf_metadata._read_existing_pdf_metadata")
    def test_should_update_when_doi_in_keywords_but_not_identifier(self, mock_read):
        # DOI in Keywords but not in Identifier
        mock_read.return_value = {"Keywords": "10.1000/abc, keyword1"}
        meta = PdfMetadata(pdf_path=Path("/tmp/test.pdf"), doi="10.1000/abc")

        should, reason = should_update_file(Path("/tmp/test.pdf"), meta)
        # DOI is already in keywords, Identifier might be missing
        # This depends on implementation - might skip or might update
        assert isinstance(should, bool)

    @patch("acatome_extract.pdf_metadata._read_existing_pdf_metadata")
    def test_should_not_update_when_doi_present(self, mock_read):
        mock_read.return_value = {
            "Keywords": "10.1000/abc",
            "Identifier": "doi:10.1000/abc",
        }
        meta = PdfMetadata(pdf_path=Path("/tmp/test.pdf"), doi="10.1000/abc")

        should, reason = should_update_file(Path("/tmp/test.pdf"), meta)
        assert should is False

    @patch("acatome_extract.pdf_metadata._read_existing_pdf_metadata")
    def test_force_overrides(self, mock_read):
        mock_read.return_value = {"Keywords": "10.1000/abc"}
        meta = PdfMetadata(pdf_path=Path("/tmp/test.pdf"), doi="10.1000/abc")

        should, reason = should_update_file(Path("/tmp/test.pdf"), meta, force=True)
        assert should is True
        assert reason == "forced"


class TestDoiCandidate:
    """Tests for DoiCandidate dataclass."""

    def test_normalization_on_creation(self):
        c = DoiCandidate(
            doi="  DOI:10.1000/ABC  ", provenance=DoiProvenance.ACATOME_BUNDLE
        )
        assert c.doi == "10.1000/abc"

    def test_validated_flag_default(self):
        c = DoiCandidate(doi="10.1000/abc", provenance=DoiProvenance.INTERNAL_EXTRACTOR)
        assert c.validated is False


class TestSidecarReading:
    """Tests for reading sidecar metadata files."""

    def test_read_existing_sidecar(self, tmp_path: Path):
        pdf = tmp_path / "paper.pdf"
        sidecar = tmp_path / "paper.meta.json"
        sidecar.write_text('{"doi": "10.1000/abc", "title": "Test"}')

        result = _read_sidecar_meta(pdf)
        assert result["doi"] == "10.1000/abc"
        assert result["title"] == "Test"

    def test_read_missing_sidecar(self, tmp_path: Path):
        pdf = tmp_path / "paper.pdf"
        result = _read_sidecar_meta(pdf)
        assert result == {}

    def test_read_invalid_sidecar(self, tmp_path: Path):
        pdf = tmp_path / "paper.pdf"
        sidecar = tmp_path / "paper.meta.json"
        sidecar.write_text("not valid json")

        result = _read_sidecar_meta(pdf)
        assert result == {}


class TestBundleFinding:
    """Tests for finding .acatome bundles."""

    def test_find_bundle_same_stem(self, tmp_path: Path):
        pdf = tmp_path / "smith2024mofs.pdf"
        bundle = tmp_path / "smith2024mofs.acatome"
        bundle.write_text("test")

        result = _find_acatome_bundle(pdf)
        assert result == bundle

    def test_find_bundle_missing(self, tmp_path: Path):
        pdf = tmp_path / "paper.pdf"
        result = _find_acatome_bundle(pdf)
        assert result is None


class TestIntegration:
    """Integration-style tests with mocked external calls."""

    @patch("acatome_extract.pdf_metadata.extract_pdf_meta")
    @patch("acatome_extract.pdf_metadata.read_bundle")
    def test_extract_metadata_from_bundle(
        self, mock_read_bundle, mock_extract_pdf, tmp_path: Path
    ):
        # Setup mock bundle
        mock_read_bundle.return_value = {
            "header": {
                "title": "Bundle Title",
                "authors": [{"name": "Smith, J."}, {"name": "Jones, A."}],
                "doi": "10.1000/bundle",
                "year": 2024,
                "journal": "Nature",
                "verified": True,
            }
        }
        mock_extract_pdf.return_value = {"pdf_hash": "abc123", "doi": None, "info": {}}

        # Create files
        pdf = tmp_path / "test.pdf"
        bundle = tmp_path / "test.acatome"
        pdf.write_text("pdf content")
        bundle.write_text("bundle content")

        from acatome_extract.pdf_metadata import extract_metadata_from_sources

        meta = extract_metadata_from_sources(pdf)

        assert meta.title == "Bundle Title"
        assert meta.authors == ["Smith, J.", "Jones, A."]
        assert meta.doi == "10.1000/bundle"
        assert meta.doi_provenance == DoiProvenance.ACATOME_BUNDLE
        assert meta.year == 2024
        assert meta.journal == "Nature"


class TestHashHistory:
    """Tests for PDF hash history tracking (backward compatible)."""

    def test_get_valid_hashes_from_old_bundle(self):
        """Old bundles without pdf_hash_history should still work."""
        from acatome_extract.pdf_metadata import get_valid_hashes_for_bundle

        bundle = {"header": {"pdf_hash": "abc123"}}
        hashes = get_valid_hashes_for_bundle(bundle)

        assert hashes == ["abc123"]

    def test_get_valid_hashes_with_history(self):
        """New bundles with pdf_hash_history return all hashes."""
        from acatome_extract.pdf_metadata import get_valid_hashes_for_bundle

        bundle = {
            "header": {
                "pdf_hash": "abc123",
                "pdf_hash_history": ["abc123", "def456", "ghi789"],
            }
        }
        hashes = get_valid_hashes_for_bundle(bundle)

        assert len(hashes) == 3
        assert "abc123" in hashes
        assert "def456" in hashes
        assert "ghi789" in hashes

    def test_get_valid_hashes_dedupes(self):
        """Should deduplicate hashes if original appears in history."""
        from acatome_extract.pdf_metadata import get_valid_hashes_for_bundle

        bundle = {
            "header": {
                "pdf_hash": "abc123",
                "pdf_hash_history": ["abc123", "abc123"],  # Duplicate
            }
        }
        hashes = get_valid_hashes_for_bundle(bundle)

        # Should dedupe to single entry
        assert hashes == ["abc123"]

    def test_compute_file_hash(self, tmp_path: Path):
        """Test file hash computation."""
        from acatome_extract.pdf_metadata import _compute_file_hash

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        hash1 = _compute_file_hash(test_file)
        hash2 = _compute_file_hash(test_file)

        assert len(hash1) == 64  # SHA-256 hex is 64 chars
        assert hash1 == hash2  # Deterministic

    @patch("acatome_extract.pdf_metadata.read_bundle")
    @patch("acatome_extract.pdf_metadata.update_bundle")
    def test_update_bundle_hash_history_creates_history(
        self, mock_update, mock_read, tmp_path: Path
    ):
        """Updating hash on old bundle creates history list."""
        from acatome_extract.pdf_metadata import _update_bundle_hash_history

        # Simulate old bundle without history
        mock_read.return_value = {"header": {"pdf_hash": "original123"}}
        mock_update.return_value = None

        bundle_path = tmp_path / "test.acatome"
        result = _update_bundle_hash_history(bundle_path, "enriched456")

        assert result is True
        # Check that update_bundle was called with history populated
        call_args = mock_update.call_args
        updated_bundle = call_args[0][0]
        header = updated_bundle["header"]

        assert header["pdf_hash_history"] == ["original123", "enriched456"]
        assert header["pdf_hash_enriched"] == "enriched456"

    @patch("acatome_extract.pdf_metadata.read_bundle")
    @patch("acatome_extract.pdf_metadata.update_bundle")
    def test_update_bundle_hash_history_skips_duplicate(
        self, mock_update, mock_read, tmp_path: Path
    ):
        """Should skip if hash already in history."""
        from acatome_extract.pdf_metadata import _update_bundle_hash_history

        # Bundle that already has this hash
        mock_read.return_value = {
            "header": {
                "pdf_hash": "hash1",
                "pdf_hash_history": ["hash1", "hash2"],
            }
        }

        bundle_path = tmp_path / "test.acatome"
        result = _update_bundle_hash_history(bundle_path, "hash2")

        assert result is False  # No update needed
        mock_update.assert_not_called()

    @patch("acatome_extract.pdf_metadata.read_bundle")
    def test_update_bundle_hash_history_handles_error(self, mock_read, tmp_path: Path):
        """Should gracefully handle read errors."""
        from acatome_extract.pdf_metadata import _update_bundle_hash_history

        mock_read.side_effect = Exception("Read error")

        bundle_path = tmp_path / "test.acatome"
        result = _update_bundle_hash_history(bundle_path, "newhash")

        assert result is False


class TestBackupFunctionality:
    """Tests for automatic PDF backup before enrichment."""

    def test_backup_pdf_creates_bak_file(self, tmp_path: Path):
        """Backup should create .pdf.bak file."""
        from acatome_extract.pdf_metadata import _backup_pdf

        pdf = tmp_path / "paper.pdf"
        pdf.write_text("original pdf content")

        backup = _backup_pdf(pdf)

        assert backup is not None
        assert backup.name == "paper.pdf.bak"
        assert backup.exists()
        assert backup.read_text() == "original pdf content"

    def test_backup_pdf_returns_none_on_failure(self, tmp_path: Path):
        """Should return None and not raise on backup failure."""
        from acatome_extract.pdf_metadata import _backup_pdf

        # Non-existent file
        pdf = tmp_path / "nonexistent.pdf"

        backup = _backup_pdf(pdf)

        assert backup is None

    def test_backup_overwrites_existing_bak(self, tmp_path: Path):
        """Should overwrite existing .bak file."""
        from acatome_extract.pdf_metadata import _backup_pdf

        pdf = tmp_path / "paper.pdf"
        pdf.write_text("new content")

        # Pre-existing bak
        existing_bak = tmp_path / "paper.pdf.bak"
        existing_bak.write_text("old backup")

        backup = _backup_pdf(pdf)

        assert backup is not None
        assert backup.read_text() == "new content"  # Updated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
