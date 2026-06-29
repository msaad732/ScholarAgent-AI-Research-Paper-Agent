"""arxiv integration: search for papers and download their PDFs."""
import os
import time

import arxiv

from config import MAX_PAPERS_PER_SEARCH, PAPERS_DIR
from utils.helpers import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_DOWNLOAD_TIMEOUT = 30  # seconds


def search_arxiv(query: str, max_results: int = MAX_PAPERS_PER_SEARCH) -> list[dict]:
    """Search arxiv for papers matching a query, sorted by relevance.

    Args:
        query: Free-text search query.
        max_results: Maximum number of papers to return.

    Returns:
        A list of paper metadata dicts. Empty on failure.
    """
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=_MAX_RETRIES)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            results: list[dict] = []
            for result in client.results(search):
                arxiv_id = result.get_short_id()
                results.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": result.title.strip(),
                        "authors": [a.name for a in result.authors],
                        "abstract": (result.summary or "").strip(),
                        "published": result.published.strftime("%Y-%m-%d")
                        if result.published
                        else "Unknown",
                        "pdf_url": result.pdf_url,
                        "categories": list(result.categories),
                    }
                )
            logger.info("arxiv search '%s' returned %d results", query, len(results))
            return results
        except Exception as exc:  # noqa: BLE001 - arxiv can raise many error types
            wait = 2 ** attempt
            logger.warning(
                "arxiv search attempt %d/%d failed: %s (retrying in %ss)",
                attempt,
                _MAX_RETRIES,
                exc,
                wait,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    logger.error("arxiv search failed for query: %s", query)
    return []


def download_paper(arxiv_id: str, pdf_url: str) -> str | None:
    """Download a paper PDF to the local papers directory, using a cache.

    Args:
        arxiv_id: The arxiv identifier, used as the file name.
        pdf_url: Direct URL to the PDF.

    Returns:
        The local file path, or None on failure.
    """
    os.makedirs(PAPERS_DIR, exist_ok=True)
    # arxiv ids may contain '/', sanitise for a flat filename
    safe_id = arxiv_id.replace("/", "_")
    local_path = os.path.join(PAPERS_DIR, f"{safe_id}.pdf")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        logger.info("Paper %s already downloaded (cache hit)", arxiv_id)
        return local_path

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            import urllib.request

            req = urllib.request.Request(pdf_url, headers={"User-Agent": "ScholarAgent/1.0"})
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as response:
                data = response.read()
            if not data:
                raise ValueError("Empty response body")
            with open(local_path, "wb") as fh:
                fh.write(data)
            logger.info("Downloaded paper %s -> %s", arxiv_id, local_path)
            return local_path
        except Exception as exc:  # noqa: BLE001 - network errors vary
            wait = 2 ** attempt
            logger.warning(
                "Download attempt %d/%d for %s failed: %s",
                attempt,
                _MAX_RETRIES,
                arxiv_id,
                exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(wait)

    logger.error("Failed to download paper %s after %d attempts", arxiv_id, _MAX_RETRIES)
    return None


def fetch_and_download(query: str, max_results: int = MAX_PAPERS_PER_SEARCH) -> list[dict]:
    """Search arxiv and download matching PDFs, returning only successful downloads.

    Args:
        query: Free-text search query.
        max_results: Maximum number of papers to fetch.

    Returns:
        A list of paper dicts, each with an added ``local_path`` key.
    """
    papers = search_arxiv(query, max_results)
    downloaded: list[dict] = []

    for paper in papers:
        local_path = download_paper(paper["arxiv_id"], paper["pdf_url"])
        if local_path:
            paper["local_path"] = local_path
            downloaded.append(paper)

    logger.info(
        "fetch_and_download '%s': %d/%d papers downloaded", query, len(downloaded), len(papers)
    )
    return downloaded
