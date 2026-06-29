"""Tool: search the indexed knowledge base for relevant excerpts."""
from langchain_core.tools import tool

from core.vector_store import VectorStoreManager
from utils.helpers import get_logger, truncate

logger = get_logger(__name__)


def make_search_tool(vector_store: VectorStoreManager):
    """Create a ``search_knowledge_base`` tool bound to a vector store.

    Args:
        vector_store: Initialised vector store manager.

    Returns:
        A LangChain tool callable.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the indexed research papers for information relevant to the query.
        Use this as your FIRST step for any question."""
        results = vector_store.query(query)
        if not results:
            return "No relevant results found in the knowledge base."

        blocks = []
        for r in results:
            meta = r["metadata"]
            tag = "YOUR PAPER" if meta.get("source_type") == "upload" else "Paper"
            ident = (
                "the user's own upload"
                if meta.get("source_type") == "upload"
                else f"arxiv:{meta.get('arxiv_id', '?')}"
            )
            blocks.append(
                f"[{tag}: {meta.get('title', 'Unknown')} | {ident} | "
                f"Section: {meta.get('section', 'Unknown')} | "
                f"relevance: {r.get('score', 0):.2f}]\n{truncate(r['text'], 700)}"
            )
        logger.info("search_knowledge_base: %d results for '%s'", len(results), query)
        return "\n\n---\n\n".join(blocks)

    return search_knowledge_base
