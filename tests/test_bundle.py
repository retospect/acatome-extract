"""Tests for bundle I/O."""

from __future__ import annotations

from acatome_extract.bundle import read_bundle, write_bundle


class TestBundle:
    def test_roundtrip(self, tmp_path):
        data = {
            "header": {"paper_id": "doi:10.1038/test", "slug": "test"},
            "blocks": [{"node_id": "doi:10.1038/test-p00-000", "text": "hello"}],
            "enrichment_meta": None,
        }
        path = tmp_path / "test.acatome"
        write_bundle(data, path)
        assert path.exists()

        loaded = read_bundle(path)
        assert loaded["header"]["paper_id"] == "doi:10.1038/test"
        assert loaded["blocks"][0]["text"] == "hello"

    def test_read_existing(self, tmp_bundle):
        data = read_bundle(tmp_bundle)
        assert data["header"]["slug"] == "smith2024quantum"
        assert len(data["blocks"]) == 1

    def test_gzipped(self, tmp_path):

        data = {"header": {"paper_id": "test"}, "blocks": [], "enrichment_meta": None}
        path = tmp_path / "test.acatome"
        write_bundle(data, path)

        # Verify it's actually gzipped
        with open(path, "rb") as f:
            magic = f.read(2)
        assert magic == b"\x1f\x8b", "File should be gzip-compressed"

    def test_creates_parent_dirs(self, tmp_path):
        data = {"header": {}, "blocks": [], "enrichment_meta": None}
        path = tmp_path / "deep" / "nested" / "test.acatome"
        write_bundle(data, path)
        assert path.exists()

    def test_unicode(self, tmp_path):
        data = {
            "header": {"title": "Über die Quantenmechanik"},
            "blocks": [],
            "enrichment_meta": None,
        }
        path = tmp_path / "unicode.acatome"
        write_bundle(data, path)
        loaded = read_bundle(path)
        assert loaded["header"]["title"] == "Über die Quantenmechanik"
