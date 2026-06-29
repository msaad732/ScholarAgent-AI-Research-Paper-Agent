"""Tests for academic-aware chunking."""
from core.pdf_parser import chunk_paper


def test_chunk_paper_metadata_and_source_type():
    text = "Abstract\n" + (
        "This is a sufficiently long sentence about the methodology and results. " * 5
    )
    chunks = chunk_paper(text, "1234.5678", "Test Title", ["A. Author"], source_type="upload")
    assert chunks
    meta = chunks[0]["metadata"]
    assert meta["arxiv_id"] == "1234.5678"
    assert meta["source_type"] == "upload"
    assert meta["title"] == "Test Title"
    assert meta["total_chunks"] == len(chunks)


def test_chunk_paper_discards_tiny_text():
    assert chunk_paper("short", "id", "t", []) == []


def test_chunk_paper_defaults_to_arxiv_source():
    text = "Introduction\n" + ("A reasonably long line of academic prose here. " * 6)
    chunks = chunk_paper(text, "2000.00001", "T", ["X"])
    assert chunks
    assert chunks[0]["metadata"]["source_type"] == "arxiv"
