"""Tests for utility helpers: text cleaning, truncation, token count, rate limiter."""
from utils.helpers import RateLimiter, clean_text, count_tokens, truncate


def test_clean_text_dehyphenates_line_breaks():
    assert clean_text("algo-\nrithm") == "algorithm"


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_clean_text_empty():
    assert clean_text("") == ""


def test_truncate_long_and_short():
    assert truncate("hello world", 5) == "hell…"
    assert truncate("hi", 5) == "hi"


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("a" * 40) >= 1


def test_rate_limiter_allows_then_blocks():
    limiter = RateLimiter(max_calls=2, window_seconds=60)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False
    assert limiter.retry_after() >= 1
