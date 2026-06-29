"""Tool: compare approaches/results across multiple papers in the knowledge base."""
from collections import defaultdict

from langchain_core.tools import tool

from core.vector_store import VectorStoreManager
from utils.helpers import get_logger, truncate

logger = get_logger(__name__)


def make_compare_tool(vector_store: VectorStoreManager):
    """Create a ``compare_papers`` tool bound to a vector store.

    Args:
        vector_store: Initialised vector store manager.

    Returns:
        A LangChain tool callable.
    """

    @tool
    def compare_papers(aspect: str) -> str:
        """Compare methodologies, approaches, or results across papers in the knowledge base.
        Input should describe what aspect to compare, e.g., 'Compare the training
        approaches used in transformer papers'."""
        # Retrieve a wider set so multiple papers are represented.
        results = vector_store.query(aspect, top_k=12)
        if not results:
            return "No relevant results found in the knowledge base to compare."

        by_paper: dict[str, list[dict]] = defaultdict(list)
        titles: dict[str, str] = {}
        for r in results:
            meta = r["metadata"]
            aid = meta.get("arxiv_id", "?")
            by_paper[aid].append(r)
            titles[aid] = meta.get("title", "Unknown")

        if len(by_paper) < 2:
            note = (
                "\n\n(Note: only one paper in the knowledge base is relevant to this "
                "aspect. Consider fetching more papers for a richer comparison.)"
            )
        else:
            note = ""

        blocks = []
        for aid, chunks in by_paper.items():
            # Use the single most relevant excerpt per paper.
            best = max(chunks, key=lambda c: c.get("score", 0))
            is_upload = best["metadata"].get("source_type") == "upload"
            label = (
                f"YOUR PAPER: {titles[aid]} (the user's own upload)"
                if is_upload
                else f"Paper: {titles[aid]} (arxiv:{aid})"
            )
            blocks.append(f"{label}\nRelevant excerpt: {truncate(best['text'], 600)}")
        logger.info("compare_papers: %d papers compared on '%s'", len(by_paper), aspect)
        return "\n---\n".join(blocks) + note

    return compare_papers
