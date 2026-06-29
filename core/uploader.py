"""Ingest a user-uploaded PDF into the knowledge base (tagged as an upload)."""
import hashlib
import os
import re
from collections import Counter

from config import PAPERS_DIR
from core.pdf_parser import chunk_paper, extract_text_from_pdf, extract_title_from_pdf
from core.vector_store import VectorStoreManager
from utils.helpers import get_logger

logger = get_logger(__name__)

# Common words to ignore when deriving an arxiv search query from a title.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "using",
    "via", "based", "approach", "method", "study", "analysis", "paper", "novel",
    "towards", "toward", "from", "by", "is", "are", "we", "our", "this", "that",
    "experimental", "empirical", "evaluation", "framework", "model", "models",
}


def _title_from_filename(filename: str) -> str:
    """Derive a human-readable title from an uploaded file name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"[_\-]+", " ", base).strip()
    return base.title() if base else "Uploaded Paper"


def ingest_uploaded_pdf(
    vector_store: VectorStoreManager, file_bytes: bytes, filename: str
) -> dict | None:
    """Save, parse, chunk, and index an uploaded PDF as an ``upload`` paper.

    Args:
        vector_store: Initialised vector store manager.
        file_bytes: Raw bytes of the uploaded PDF.
        filename: Original file name (used to derive a title).

    Returns:
        A dict with ``paper_id``, ``title``, ``num_chunks``, ``text`` and
        ``local_path`` keys, or None if the PDF could not be read.
    """
    if not file_bytes:
        return None

    os.makedirs(PAPERS_DIR, exist_ok=True)
    short_hash = hashlib.sha1(file_bytes).hexdigest()[:10]
    paper_id = f"upload_{short_hash}"
    local_path = os.path.join(PAPERS_DIR, f"{paper_id}.pdf")

    # Cache: write the file once.
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
        try:
            with open(local_path, "wb") as fh:
                fh.write(file_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save uploaded PDF: %s", exc)
            return None

    text = extract_text_from_pdf(local_path)
    if not text:
        logger.warning("No text extracted from uploaded PDF %s", filename)
        return None

    # Prefer the real title detected from the PDF; fall back to the filename.
    title = extract_title_from_pdf(local_path) or _title_from_filename(filename)

    # Index only if not already present (re-uploading the same file is a no-op).
    num_chunks = 0
    if not vector_store.is_paper_indexed(paper_id):
        chunks = chunk_paper(text, paper_id, title, ["(you)"], source_type="upload")
        num_chunks = vector_store.add_paper(chunks)

    logger.info("Ingested upload '%s' as %s (%d chunks)", title, paper_id, num_chunks)
    return {
        "paper_id": paper_id,
        "title": title,
        "num_chunks": num_chunks,
        "text": text,
        "local_path": local_path,
    }


def derive_search_query(title: str, text: str, max_terms: int = 6) -> str:
    """Build an arxiv search query from a paper's title and salient keywords.

    Args:
        title: The paper title.
        text: The paper's full text (used to surface frequent keywords).
        max_terms: Maximum number of keyword terms to include.

    Returns:
        A search query string suitable for ``paper_fetcher.search_arxiv``.
    """
    title_terms = [
        w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", title.lower())
        if w not in _STOPWORDS
    ]

    # Supplement with the most frequent meaningful words from the body.
    body_words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        if w not in _STOPWORDS
    ]
    common = [w for w, _ in Counter(body_words).most_common(40)]

    terms: list[str] = []
    for term in title_terms + common:
        if term not in terms:
            terms.append(term)
        if len(terms) >= max_terms:
            break

    query = " ".join(terms) if terms else title
    logger.info("Derived arxiv query: '%s'", query)
    return query
