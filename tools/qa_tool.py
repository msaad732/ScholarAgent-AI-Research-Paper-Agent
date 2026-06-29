"""Tool: fetch new papers from arxiv and index them into the knowledge base."""
from langchain_core.tools import tool

from config import MAX_PAPERS_PER_SEARCH
from core import paper_fetcher
from core.pdf_parser import chunk_paper, extract_text_from_pdf
from core.vector_store import VectorStoreManager
from utils.helpers import get_logger

logger = get_logger(__name__)


def make_fetch_tool(vector_store: VectorStoreManager):
    """Create a ``fetch_new_papers`` tool bound to a vector store.

    Args:
        vector_store: Initialised vector store manager.

    Returns:
        A LangChain tool callable.
    """

    @tool
    def fetch_new_papers(query: str) -> str:
        """Search arxiv for new papers on the given topic and add them to the knowledge base.
        Use this when the existing knowledge base doesn't have enough information."""
        papers = paper_fetcher.fetch_and_download(query, MAX_PAPERS_PER_SEARCH)
        if not papers:
            return "No papers found on arxiv for this query."

        indexed_titles: list[str] = []
        for paper in papers:
            try:
                if vector_store.is_paper_indexed(paper["arxiv_id"]):
                    indexed_titles.append(f"{paper['title']} (already indexed)")
                    continue
                text = extract_text_from_pdf(paper["local_path"])
                if not text:
                    logger.warning("No text extracted for %s", paper["arxiv_id"])
                    continue
                chunks = chunk_paper(
                    text, paper["arxiv_id"], paper["title"], paper["authors"]
                )
                added = vector_store.add_paper(chunks)
                if added:
                    indexed_titles.append(paper["title"])
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to index %s: %s", paper.get("arxiv_id"), exc)

        if not indexed_titles:
            return "Papers were found but could not be indexed (parsing failed)."

        listing = "\n".join(f"- {t}" for t in indexed_titles)
        logger.info("fetch_new_papers indexed %d papers for '%s'", len(indexed_titles), query)
        return f"Fetched and indexed {len(indexed_titles)} new papers:\n{listing}"

    return fetch_new_papers
