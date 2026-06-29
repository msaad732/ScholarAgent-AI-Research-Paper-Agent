"""Cross-encoder reranking to improve retrieval precision.

A bi-encoder (the embedding model) is fast but coarse; a cross-encoder scores
each (query, chunk) pair jointly and is far more accurate at ordering. We fetch a
wider candidate set from ChromaDB, then rerank it down to the final top-k.
"""
from config import RERANK_MODEL
from utils.helpers import get_logger

logger = get_logger(__name__)

_reranker = None


def get_reranker():
    """Lazily load and cache the cross-encoder model (downloaded on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model %s", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score candidate chunks against the query and return the best ``top_k``.

    Falls back to the original ordering (truncated) if the model is unavailable.

    Args:
        query: The search query.
        chunks: Candidate chunk dicts (each with a ``text`` key).
        top_k: Number of chunks to return.

    Returns:
        The reranked top-k chunks, each annotated with a ``rerank_score``.
    """
    if not chunks:
        return []
    try:
        model = get_reranker()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]
    except Exception as exc:  # noqa: BLE001 - never let reranking break retrieval
        logger.error("Reranking failed, using original order: %s", exc)
        return chunks[:top_k]
