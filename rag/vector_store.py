from typing import Any
import faiss
import numpy as np


class VectorStore:
    """In-memory FAISS vector store using IndexFlatIP (Inner Product) for normalized cosine similarity."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata_records: list[dict[str, Any]]) -> None:
        """Adds normalized embedding vectors and parallel metadata records to the store.

        Args:
            embeddings: 2D numpy array of shape (N, dimension).
            metadata_records: List of dictionary metadata records parallel to embeddings.
        """
        if embeddings.size == 0 or len(metadata_records) == 0:
            return

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)
        self.metadata.extend(metadata_records)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.35
    ) -> list[dict[str, Any]]:
        """Searches FAISS index for top-k similar vectors matching or exceeding score threshold.

        Args:
            query_vector: 2D or 1D numpy array representing query embedding.
            k: Maximum number of top results to return.
            threshold: Minimum similarity score cutoff (cosine similarity).

        Returns:
            List of matching metadata dicts with 'score' field included, sorted descending by score.
        """
        if self.index.ntotal == 0 or query_vector.size == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_f32 = query_vector.astype(np.float32)
        actual_k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_f32, actual_k)

        results: list[dict[str, Any]] = []
        if len(scores) == 0 or len(indices) == 0:
            return []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            score_val = float(score)
            if score_val >= threshold:
                record = dict(self.metadata[idx])
                record["score"] = round(score_val, 4)
                results.append(record)

        # Sort results descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def reset(self) -> None:
        """Clears FAISS index and metadata storage."""
        self.index.reset()
        self.metadata = []


def create_index(embeddings: np.ndarray, dimension: int = 384) -> faiss.IndexFlatIP:
    """Creates a FAISS IndexFlatIP index and populates it with normalized embeddings."""
    dim = embeddings.shape[1] if embeddings.ndim > 1 else dimension
    index = faiss.IndexFlatIP(dim)
    if embeddings.size > 0:
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        index.add(embeddings.astype(np.float32))
    return index


def search(
    index: Any,
    query_embedding: np.ndarray,
    chunks: list[dict[str, Any]],
    k: int = 5,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Searches a FAISS index and returns matching chunks with scores."""
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    if index.ntotal == 0 or query_embedding.size == 0 or not chunks:
        return []

    actual_k = min(k, index.ntotal, len(chunks))
    scores, indices = index.search(query_embedding.astype(np.float32), actual_k)

    results = []
    if len(scores) == 0 or len(indices) == 0:
        return []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        score_val = float(score)
        if score_val >= threshold:
            item = dict(chunks[idx])
            item["score"] = round(score_val, 4)
            results.append(item)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results