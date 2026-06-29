"""Core RAG: retrieve relevant chunks and generate grounded answers via Groq."""
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RESULTS
from core.vector_store import VectorStoreManager
from utils.helpers import get_logger

logger = get_logger(__name__)

_SYSTEM_INSTRUCTION = (
    "You are a research assistant. Answer based ONLY on the provided paper "
    "excerpts. Cite papers by their arxiv ID and title. If the context doesn't "
    "contain enough information, say so clearly."
)


def _get_llm():
    """Construct a ChatGroq client, raising a clear error if the key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Assemble a grounded RAG prompt from retrieved chunks.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks with ``text`` and ``metadata``.

    Returns:
        A fully-formatted prompt string.
    """
    if context_chunks:
        formatted = []
        for chunk in context_chunks:
            meta = chunk["metadata"]
            formatted.append(
                f"--- Paper: {meta.get('title', 'Unknown')} "
                f"(arxiv:{meta.get('arxiv_id', '?')}) | "
                f"Section: {meta.get('section', 'Unknown')} ---\n"
                f"{chunk['text']}"
            )
        context_block = "\n\n".join(formatted)
    else:
        context_block = "(No relevant excerpts were found in the knowledge base.)"

    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"=== PAPER EXCERPTS ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ==="
    )


def query_and_respond(
    query: str, vector_store: VectorStoreManager, filter_dict: dict | None = None
) -> dict:
    """Retrieve context and generate a grounded answer to a question.

    Args:
        query: The user's question.
        vector_store: Initialised vector store manager.
        filter_dict: Optional metadata filter for retrieval.

    Returns:
        A dict with ``answer``, ``sources`` and ``num_sources`` keys.
    """
    chunks = vector_store.query(query, top_k=TOP_K_RESULTS, filter_dict=filter_dict)

    sources = [
        {
            "arxiv_id": c["metadata"].get("arxiv_id", "?"),
            "title": c["metadata"].get("title", "Unknown"),
            "section": c["metadata"].get("section", "Unknown"),
            "relevance_score": round(c.get("score", 0.0), 3),
        }
        for c in chunks
    ]

    prompt = build_rag_prompt(query, chunks)

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM generation failed: %s", exc)
        answer = f"An error occurred while generating the answer: {exc}"

    return {"answer": answer, "sources": sources, "num_sources": len(sources)}


def stream_answer(
    query: str, vector_store: VectorStoreManager, filter_dict: dict | None = None
):
    """Retrieve context and stream a grounded answer token-by-token.

    Args:
        query: The user's question.
        vector_store: Initialised vector store manager.
        filter_dict: Optional metadata filter for retrieval.

    Returns:
        A ``(generator, sources)`` tuple. The generator yields answer text
        chunks; ``sources`` is the list of cited chunks (known before
        generation, since retrieval happens first).
    """
    chunks = vector_store.query(query, top_k=TOP_K_RESULTS, filter_dict=filter_dict)
    sources = [
        {
            "arxiv_id": c["metadata"].get("arxiv_id", "?"),
            "title": c["metadata"].get("title", "Unknown"),
            "section": c["metadata"].get("section", "Unknown"),
            "snippet": c["text"][:500],
            "is_own": c["metadata"].get("source_type") == "upload",
        }
        for c in chunks
    ]
    prompt = build_rag_prompt(query, chunks)

    def _generator():
        try:
            for piece in _get_llm().stream(prompt):
                yield piece.content if hasattr(piece, "content") else str(piece)
        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming generation failed: %s", exc)
            yield f"\n\n_An error occurred while generating the answer: {exc}_"

    return _generator(), sources
