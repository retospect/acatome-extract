"""Tests for acatome_extract.chunker."""

from acatome_extract.chunker import split_text


class TestSplitText:
    def test_short_text_unchanged(self):
        result = split_text("Hello world.", chunk_size=100)
        assert result == ["Hello world."]

    def test_empty_text(self):
        assert split_text("") == []
        assert split_text("   ") == []

    def test_splits_on_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = split_text(text, chunk_size=20, chunk_overlap=0)
        assert len(result) >= 2
        assert "Para one." in result[0]

    def test_respects_chunk_size(self):
        text = " ".join(["word"] * 500)  # ~2500 chars
        result = split_text(text, chunk_size=200, chunk_overlap=0)
        for chunk in result:
            assert len(chunk) <= 200 + 10  # small tolerance for separator

    def test_overlap_present(self):
        # Build text with many small paragraphs that force multiple chunks
        paras = [f"Paragraph {i} content here." for i in range(20)]
        text = "\n\n".join(paras)
        result = split_text(text, chunk_size=200, chunk_overlap=50)
        assert len(result) >= 2
        # Some overlap text from end of chunk N should appear in chunk N+1
        for i in range(len(result) - 1):
            tail = result[i][-50:]
            # At least part of the tail should appear in the next chunk
            # (overlap means shared content)
            assert any(
                word in result[i + 1] for word in tail.split() if len(word) > 3
            ), f"No overlap between chunk {i} and {i + 1}"

    def test_sentence_splitting(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = split_text(text, chunk_size=40, chunk_overlap=0)
        assert len(result) >= 2

    def test_long_word_not_split(self):
        # A single word longer than chunk_size should be kept whole
        text = "x" * 1000
        result = split_text(text, chunk_size=200, chunk_overlap=0)
        assert len(result) == 1
        assert result[0] == "x" * 1000

    def test_academic_text(self):
        text = (
            "1. Introduction\n\n"
            "We present a novel approach to quantum error correction. "
            "Our method combines surface codes with machine learning "
            "to achieve fault-tolerant quantum computation.\n\n"
            "2. Methods\n\n"
            "The experimental setup consists of a superconducting "
            "quantum processor with 127 qubits arranged in a heavy-hex "
            "lattice topology.\n\n"
            "3. Results\n\n"
            "We observe a significant reduction in logical error rates "
            "compared to previous approaches."
        )
        result = split_text(text, chunk_size=200, chunk_overlap=50)
        assert len(result) >= 3
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in result)
