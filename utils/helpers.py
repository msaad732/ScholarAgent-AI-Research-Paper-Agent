"""Shared helpers: logging, text cleaning, token counting, and rate limiting."""
import logging
import re
import threading
import time
from collections import deque

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "scholaragent") -> logging.Logger:
    """Return a configured logger, creating it once per name.

    Args:
        name: Logger name, usually the module's ``__name__``.

    Returns:
        A logging.Logger writing to the console at INFO level.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    _LOGGERS[name] = logger
    return logger


def clean_text(text: str) -> str:
    """Normalise whitespace and repair hyphenated line breaks in extracted text.

    Args:
        text: Raw text, typically from a PDF.

    Returns:
        Cleaned text with collapsed whitespace and de-hyphenated words.
    """
    if not text:
        return ""

    # Fix words split across line breaks: "algo-\nrithm" -> "algorithm"
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalise Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Estimate token count using a simple character heuristic (~4 chars/token).

    Args:
        text: Text to measure.

    Returns:
        Approximate number of tokens.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate(text: str, max_chars: int = 60) -> str:
    """Truncate a string to ``max_chars`` characters with an ellipsis.

    Args:
        text: Input string.
        max_chars: Maximum length before truncation.

    Returns:
        The original string, or a truncated version ending in ``…``.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks call timestamps within a fixed window and rejects calls once the
    maximum is reached. Used to throttle abusive usage on public deployments.
    """

    def __init__(self, max_calls: int, window_seconds: int) -> None:
        """Initialise the limiter.

        Args:
            max_calls: Maximum number of calls allowed per window.
            window_seconds: Length of the sliding window, in seconds.
        """
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        """Drop timestamps that have fallen outside the current window."""
        cutoff = now - self.window
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def allow(self) -> bool:
        """Record and allow a call if under the limit, otherwise reject it.

        Returns:
            True if the call is permitted, False if the limit is exceeded.
        """
        now = time.time()
        with self._lock:
            self._evict(now)
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True

    def retry_after(self) -> int:
        """Return seconds until the oldest in-window call expires.

        Returns:
            Seconds to wait before another call is likely to be allowed (0 if free).
        """
        now = time.time()
        with self._lock:
            self._evict(now)
            if len(self._calls) < self.max_calls:
                return 0
            return max(1, int(self.window - (now - self._calls[0])))
