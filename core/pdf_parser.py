"""PDF text extraction and academic-aware chunking for research papers."""
import re
from collections import Counter

import fitz  # PyMuPDF

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from utils.helpers import clean_text, get_logger

logger = get_logger(__name__)

# Patterns that typically mark the start of a section in an academic paper.
_SECTION_PATTERNS = [
    r"^\s*abstract\s*$",
    r"^\s*\d+\.?\s+[A-Z][A-Za-z].{0,60}$",          # "1 Introduction", "2. Related Work"
    r"^\s*\d+\.\d+\.?\s+[A-Z][A-Za-z].{0,60}$",      # "3.1 Model Architecture"
    r"^\s*#{1,4}\s+.{1,60}$",                          # markdown-style "## Methodology"
    r"^\s*(introduction|related work|background|methodology|methods|approach|"
    r"experiments?|results|evaluation|discussion|conclusions?|references|"
    r"acknowledgments?|appendix)\s*$",
]
_SECTION_RE = re.compile("|".join(f"({p})" for p in _SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from a PDF file, page by page.

    Removes headers/footers that repeat across most pages, repairs hyphenated
    line breaks, and collapses redundant whitespace.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        The cleaned full text, or an empty string if extraction fails.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - corrupt/missing files
        logger.error("Could not open PDF %s: %s", pdf_path, exc)
        return ""

    try:
        pages = [page.get_text("text") for page in doc]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed extracting text from %s: %s", pdf_path, exc)
        return ""
    finally:
        doc.close()

    pages = _strip_repeating_lines(pages)
    full_text = "\n\n".join(pages)
    cleaned = clean_text(full_text)
    logger.info("Extracted %d chars from %s", len(cleaned), pdf_path)
    return cleaned


def extract_title_from_pdf(pdf_path: str) -> str | None:
    """Guess a paper's title from the largest-font text near the top of page 1.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        The detected title, or None if it cannot be determined.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not open PDF for title detection %s: %s", pdf_path, exc)
        return None

    try:
        page = doc[0]
        data = page.get_text("dict")
        page_height = page.rect.height  # read BEFORE closing the document
    except Exception as exc:  # noqa: BLE001
        logger.error("Title detection failed for %s: %s", pdf_path, exc)
        return None
    finally:
        doc.close()

    # Collect (font_size, text, y_position) for every span on page 1.
    spans: list[tuple[float, str, float]] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if txt:
                    spans.append((round(span.get("size", 0), 1), txt, span.get("bbox", [0, 0])[1]))

    if not spans:
        return None

    # The title is usually the largest font on the page; join spans at that size,
    # in reading order, but only those in the top half of the page.
    max_size = max(s[0] for s in spans)
    page_height = page_height or 1000
    title_parts = [
        txt
        for size, txt, y in sorted(spans, key=lambda s: s[2])
        if size >= max_size - 0.5 and y < page_height * 0.5
    ]
    title = " ".join(title_parts).strip()
    title = re.sub(r"\s+", " ", title)

    # Reject implausible titles (too short, too long, or arxiv banners).
    if not (10 <= len(title) <= 250) or title.lower().startswith("arxiv"):
        return None
    return title


def _strip_repeating_lines(pages: list[str]) -> list[str]:
    """Remove short lines (headers/footers) that repeat across most pages."""
    if len(pages) < 3:
        return pages

    line_counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            if len(line) < 80:  # only consider short lines as candidate headers/footers
                line_counts[line] += 1

    threshold = max(3, int(len(pages) * 0.6))
    repeating = {line for line, count in line_counts.items() if count >= threshold}
    if not repeating:
        return pages

    cleaned_pages = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in repeating]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_name, section_text) tuples using header detection."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("Unknown", text)]

    sections: list[tuple[str, str]] = []
    # Capture any preamble before the first detected header.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Header", preamble))

    for i, match in enumerate(matches):
        name = match.group(0).strip().lstrip("#").strip() or "Unknown"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Character-based splitter with overlap, preferring paragraph/sentence breaks."""
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to break on a paragraph, then sentence, then space boundary.
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(sep)
                if idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_paper(
    text: str,
    arxiv_id: str,
    title: str,
    authors: list[str],
    source_type: str = "arxiv",
) -> list[dict]:
    """Split paper text into academic-aware chunks with rich metadata.

    Strategy: split by detected section headers, then by paragraphs for long
    sections, falling back to overlapping recursive character splitting.

    Args:
        text: Cleaned full text of the paper.
        arxiv_id: arxiv identifier (or a synthetic ``upload_<hash>`` id).
        title: Paper title.
        authors: List of author names.
        source_type: Origin of the paper — ``"arxiv"`` or ``"upload"``.

    Returns:
        A list of chunk dicts with ``text`` and ``metadata`` keys.
    """
    if not text:
        return []

    authors_str = ", ".join(authors) if authors else "Unknown"
    raw_chunks: list[tuple[str, str]] = []  # (section_name, chunk_text)

    for section_name, section_text in _split_by_sections(text):
        if len(section_text) <= CHUNK_SIZE:
            raw_chunks.append((section_name, section_text))
            continue
        # Section too long: split by paragraphs first.
        for para in section_text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= CHUNK_SIZE:
                raw_chunks.append((section_name, para))
            else:
                for piece in _recursive_split(para, CHUNK_SIZE, CHUNK_OVERLAP):
                    raw_chunks.append((section_name, piece))

    # Discard tiny chunks.
    raw_chunks = [(s, t) for s, t in raw_chunks if len(t) >= MIN_CHUNK_SIZE]

    total = len(raw_chunks)
    chunks: list[dict] = []
    for idx, (section_name, chunk_text) in enumerate(raw_chunks):
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors_str,
                    "section": section_name,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "source": f"{source_type}:{arxiv_id}",
                    "source_type": source_type,
                },
            }
        )

    logger.info("Chunked paper %s into %d chunks", arxiv_id, total)
    return chunks
