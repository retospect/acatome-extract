"""Tests for marker.py post-processing: junk detection and link stripping."""

from __future__ import annotations

import pytest

from acatome_extract.marker import (
    _JUNK_HEADING_RE,
    _MD_LINK_RE,
    _mark_junk,
)


class TestJunkHeadingRegex:
    @pytest.mark.parametrize(
        "text",
        [
            "OPEN ACCESS",
            "Open Access",
            "OPEN  ACCESS",
            "COPYRIGHT",
            "Copyright",
            "CITATION",
            "REVIEWED BY Yashwant Bisht, Uttaranchal University, India",
            "Reviewed by John Smith",
            "EDITED BY Jane Doe",
            "Edited by Someone",
            "*CORRESPONDENCE",
            "CORRESPONDENCE",
            "Correspondence",
            "RECEIVED 12 January 2025",
            "ACCEPTED 5 March 2025",
            "PUBLISHED 10 March 2025",
            "HANDLING EDITOR John",
            "ASSOCIATE EDITOR Jane",
            "TYPE Original Research",
        ],
    )
    def test_matches_junk(self, text):
        assert _JUNK_HEADING_RE.match(text), f"Expected match: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "Introduction",
            "1. Methods",
            "Results and Discussion",
            "Acknowledgments",
            "References",
            "Abstract",
            "Experimental Procedures",
            "2.1 Surface Characterization",
        ],
    )
    def test_does_not_match_real_headings(self, text):
        assert not _JUNK_HEADING_RE.match(text), f"Should not match: {text!r}"


class TestMarkJunk:
    def test_junk_heading_and_followers(self):
        blocks = [
            {"type": "section_header", "text": "OPEN ACCESS", "section_path": ["OPEN ACCESS"]},
            {"type": "text", "text": "Some boilerplate.", "section_path": ["OPEN ACCESS"]},
            {"type": "section_header", "text": "Introduction", "section_path": ["Introduction"]},
            {"type": "text", "text": "Real content.", "section_path": ["Introduction"]},
        ]
        result = _mark_junk(blocks)
        assert result[0]["type"] == "junk"
        assert result[1]["type"] == "junk"
        assert result[2]["type"] == "section_header"
        assert result[3]["type"] == "text"

    def test_multiple_junk_sections(self):
        blocks = [
            {"type": "section_header", "text": "OPEN ACCESS", "section_path": ["OPEN ACCESS"]},
            {"type": "text", "text": "OA text.", "section_path": ["OPEN ACCESS"]},
            {"type": "section_header", "text": "COPYRIGHT", "section_path": ["COPYRIGHT"]},
            {"type": "text", "text": "Copyright text.", "section_path": ["COPYRIGHT"]},
            {"type": "section_header", "text": "CITATION", "section_path": ["CITATION"]},
            {"type": "text", "text": "Citation text.", "section_path": ["CITATION"]},
            {"type": "section_header", "text": "Introduction", "section_path": ["Introduction"]},
            {"type": "text", "text": "Actual content.", "section_path": ["Introduction"]},
        ]
        result = _mark_junk(blocks)
        for i in range(6):
            assert result[i]["type"] == "junk", f"Block {i} should be junk"
        assert result[6]["type"] == "section_header"
        assert result[7]["type"] == "text"

    def test_no_junk_when_no_frontmatter(self):
        blocks = [
            {"type": "section_header", "text": "Introduction", "section_path": ["Introduction"]},
            {"type": "text", "text": "Content.", "section_path": ["Introduction"]},
            {"type": "section_header", "text": "Methods", "section_path": ["Methods"]},
            {"type": "text", "text": "More content.", "section_path": ["Methods"]},
        ]
        result = _mark_junk(blocks)
        assert all(b["type"] != "junk" for b in result)

    def test_junk_does_not_leak_past_real_heading(self):
        blocks = [
            {"type": "section_header", "text": "REVIEWED BY Someone", "section_path": ["REVIEWED BY Someone"]},
            {"type": "text", "text": "Reviewer info.", "section_path": ["REVIEWED BY Someone"]},
            {"type": "section_header", "text": "1. Introduction", "section_path": ["1. Introduction"]},
            {"type": "text", "text": "Real intro.", "section_path": ["1. Introduction"]},
            {"type": "text", "text": "More intro.", "section_path": ["1. Introduction"]},
        ]
        result = _mark_junk(blocks)
        assert result[0]["type"] == "junk"
        assert result[1]["type"] == "junk"
        assert result[2]["type"] == "section_header"
        assert result[3]["type"] == "text"
        assert result[4]["type"] == "text"

    def test_preserves_block_text(self):
        """Junk detection changes type but preserves text."""
        blocks = [
            {"type": "section_header", "text": "COPYRIGHT", "section_path": ["COPYRIGHT"]},
            {"type": "text", "text": "© 2025 Authors", "section_path": ["COPYRIGHT"]},
        ]
        result = _mark_junk(blocks)
        assert result[0]["text"] == "COPYRIGHT"
        assert result[1]["text"] == "© 2025 Authors"


class TestMdLinkStrip:
    def test_strip_link(self):
        text = "[Experimental study](https://example.com/article)"
        assert _MD_LINK_RE.sub(r"\1", text) == "Experimental study"

    def test_strip_multiple_links(self):
        text = "[Part one](http://a.com) and [Part two](http://b.com)"
        assert _MD_LINK_RE.sub(r"\1", text) == "Part one and Part two"

    def test_no_links_unchanged(self):
        text = "Plain heading text"
        assert _MD_LINK_RE.sub(r"\1", text) == text
