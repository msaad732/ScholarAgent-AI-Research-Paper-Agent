"""Tests for citation harvesting from tool-result strings."""
from core.agent import _harvest_sources


def test_harvest_bracket_arxiv_source():
    text = (
        "[Paper: Attention Is All You Need | arxiv:1706.03762 | "
        "Section: Introduction | relevance: 0.80]\n"
        "The Transformer is a model architecture relying on attention."
    )
    sources: dict = {}
    _harvest_sources(text, sources)
    assert "1706.03762" in sources
    src = sources["1706.03762"]
    assert src["title"] == "Attention Is All You Need"
    assert src["section"] == "Introduction"
    assert src["is_own"] is False
    assert "Transformer" in src["snippet"]


def test_harvest_your_paper_marked_as_own():
    text = (
        "[YOUR PAPER: My Draft | the user's own upload | "
        "Section: Methods | relevance: 0.50]\n"
        "We propose a new caching method."
    )
    sources: dict = {}
    _harvest_sources(text, sources)
    own = [v for v in sources.values() if v["is_own"]]
    assert own and own[0]["title"] == "My Draft"


def test_harvest_compare_format():
    text = "Paper: Some Title (arxiv:2101.00001)\nRelevant excerpt: A useful finding."
    sources: dict = {}
    _harvest_sources(text, sources)
    assert "2101.00001" in sources
