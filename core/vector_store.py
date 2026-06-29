"""ChromaDB wrapper: embed chunks locally, store them, and query by similarity."""
import os

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from core.reranker import rerank
from utils.helpers import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manage a persistent ChromaDB collection of research-paper chunks."""

    def __init__(self) -> None:
        """Initialise the persistent client, embedding function, and collection."""
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        # Local sentence-transformers embeddings — no API key required.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStoreManager ready (collection=%s)", COLLECTION_NAME)

    def add_paper(self, chunks: list[dict]) -> int:
        """Add a paper's chunks to the collection, skipping already-indexed papers.

        Args:
            chunks: Chunk dicts produced by ``pdf_parser.chunk_paper``.

        Returns:
            Number of chunks added (0 if the paper was already indexed or empty).
        """
        if not chunks:
            return 0

        arxiv_id = chunks[0]["metadata"]["arxiv_id"]
        if self.is_paper_indexed(arxiv_id):
            logger.info("Paper %s already indexed, skipping", arxiv_id)
            return 0

        ids = [f"{arxiv_id}::{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to add paper %s: %s", arxiv_id, exc)
            return 0

        logger.info("Added %d chunks for paper %s", len(ids), arxiv_id)
        return len(ids)

    def is_paper_indexed(self, arxiv_id: str) -> bool:
        """Return True if any chunk for the given arxiv id exists in the store.

        Args:
            arxiv_id: arxiv identifier to look up.

        Returns:
            True if indexed, otherwise False.
        """
        try:
            result = self.collection.get(where={"arxiv_id": arxiv_id}, limit=1)
            return bool(result and result.get("ids"))
        except Exception as exc:  # noqa: BLE001
            logger.error("is_paper_indexed failed for %s: %s", arxiv_id, exc)
            return False

    def query(
        self, query_text: str, top_k: int = TOP_K_RESULTS, filter_dict: dict | None = None
    ) -> list[dict]:
        """Query the collection by similarity, filtering low-relevance results.

        Args:
            query_text: Natural-language query.
            top_k: Number of chunks to retrieve.
            filter_dict: Optional Chroma ``where`` metadata filter.

        Returns:
            A list of dicts with ``text``, ``metadata``, ``distance`` and
            ``score`` keys, ordered by relevance.
        """
        if self.collection.count() == 0:
            return []

        # Over-fetch candidates when reranking so the cross-encoder has room to work.
        n_fetch = max(top_k, RERANK_CANDIDATES) if RERANK_ENABLED else top_k
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_fetch,
                where=filter_dict or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Query failed: %s", exc)
            return []

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        output: list[dict] = []
        for text, meta, dist in zip(docs, metas, dists):
            # Cosine distance in [0, 2]; relevance score = 1 - distance.
            score = 1.0 - float(dist)
            if score < SIMILARITY_THRESHOLD:
                continue
            output.append(
                {"text": text, "metadata": meta, "distance": float(dist), "score": score}
            )

        if RERANK_ENABLED and output:
            output = rerank(query_text, output, top_k)
        else:
            output = output[:top_k]

        logger.info("Query returned %d chunks (rerank=%s)", len(output), RERANK_ENABLED)
        return output

    def get_all_papers(self) -> list[dict]:
        """Return a deduplicated list of papers currently in the store.

        Returns:
            A list of dicts with ``arxiv_id``, ``title``, ``authors`` and
            ``total_chunks`` keys.
        """
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.error("get_all_papers failed: %s", exc)
            return []

        papers: dict[str, dict] = {}
        for meta in result.get("metadatas") or []:
            aid = meta.get("arxiv_id")
            if not aid:
                continue
            entry = papers.setdefault(
                aid,
                {
                    "arxiv_id": aid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "total_chunks": meta.get("total_chunks", 0),
                    "source_type": meta.get("source_type", "arxiv"),
                },
            )
            # Count actual chunks present in case total_chunks metadata is stale.
            entry.setdefault("_seen", 0)
        # Recount chunks accurately.
        for meta in result.get("metadatas") or []:
            aid = meta.get("arxiv_id")
            if aid in papers:
                papers[aid]["_seen"] += 1
        for entry in papers.values():
            entry["total_chunks"] = entry.pop("_seen", entry.get("total_chunks", 0))

        return list(papers.values())

    def get_paper_text(self, arxiv_id: str) -> str:
        """Reconstruct a paper's full text from its stored chunks, in order.

        Args:
            arxiv_id: Identifier of the paper to reassemble.

        Returns:
            The concatenated chunk text, or an empty string if not found.
        """
        try:
            result = self.collection.get(
                where={"arxiv_id": arxiv_id}, include=["documents", "metadatas"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("get_paper_text failed for %s: %s", arxiv_id, exc)
            return ""

        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ordered = sorted(
            zip(docs, metas), key=lambda pair: pair[1].get("chunk_index", 0)
        )
        return "\n\n".join(doc for doc, _ in ordered)

    def delete_paper(self, arxiv_id: str) -> bool:
        """Delete all chunks for a paper.

        Args:
            arxiv_id: arxiv identifier to remove.

        Returns:
            True if the paper existed and was removed, otherwise False.
        """
        if not self.is_paper_indexed(arxiv_id):
            return False
        try:
            self.collection.delete(where={"arxiv_id": arxiv_id})
            logger.info("Deleted paper %s", arxiv_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("delete_paper failed for %s: %s", arxiv_id, exc)
            return False

    def get_collection_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            A dict with ``total_chunks``, ``total_papers`` and
            ``collection_name`` keys.
        """
        try:
            total_chunks = self.collection.count()
        except Exception as exc:  # noqa: BLE001
            logger.error("get_collection_stats failed: %s", exc)
            total_chunks = 0
        return {
            "total_chunks": total_chunks,
            "total_papers": len(self.get_all_papers()),
            "collection_name": COLLECTION_NAME,
        }
