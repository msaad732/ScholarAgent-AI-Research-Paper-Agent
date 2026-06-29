"""Tool: extract numerical metrics and key quantitative findings from papers."""
import re

from langchain_core.tools import tool

from core.vector_store import VectorStoreManager
from utils.helpers import get_logger, truncate

logger = get_logger(__name__)

# Heuristics for spotting chunks that carry quantitative results.
_METRIC_RE = re.compile(
    r"(\d+\.?\d*\s*%)"                      # percentages
    r"|(\b\d+\.\d+\b)"                       # decimals
    r"|(\baccuracy\b|\bf1\b|\bbleu\b|\brouge\b|\bprecision\b|\brecall\b"
    r"|\bperplexity\b|\bmAP\b|\bAUC\b|\bSOTA\b|\bstate[- ]of[- ]the[- ]art\b)",
    re.IGNORECASE,
)


def make_extract_tool(vector_store: VectorStoreManager):
    """Create an ``extract_metrics`` tool bound to a vector store.

    Args:
        vector_store: Initialised vector store manager.

    Returns:
        A LangChain tool callable.
    """

    @tool
    def extract_metrics(query: str) -> str:
        """Extract specific numerical results, metrics, accuracy scores, or key findings
        from the indexed papers. Input should describe what metrics to find, e.g.,
        'accuracy scores for image classification models'."""
        results = vector_store.query(query, top_k=12)
        if not results:
            return "No relevant results found in the knowledge base."

        metric_blocks = []
        for r in results:
            if not _METRIC_RE.search(r["text"]):
                continue
            meta = r["metadata"]
            metric_blocks.append(
                f"[{meta.get('title', 'Unknown')} | arxiv:{meta.get('arxiv_id', '?')} | "
                f"Section: {meta.get('section', 'Unknown')}]\n{truncate(r['text'], 700)}"
            )

        if not metric_blocks:
            return (
                "No explicit numerical metrics were found in the retrieved excerpts. "
                "The relevant papers may not report quantitative results for this query."
            )

        logger.info("extract_metrics: %d metric-bearing excerpts for '%s'", len(metric_blocks), query)
        return "\n\n---\n\n".join(metric_blocks)

    return extract_metrics
