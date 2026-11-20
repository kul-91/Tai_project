from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generates normalized vector embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encodes list of text strings into normalized float32 embedding vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            2D numpy array of shape (N, D) with float32 type.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a normalized float32 vector.

        Args:
            query: Query string to embed.

        Returns:
            2D numpy array of shape (1, D) with float32 type.
        """
        if not query or not query.strip():
            return np.empty((0, 384), dtype=np.float32)

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return np.array(embedding, dtype=np.float32)


def create_embeddings(
    chunks: Any,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Convenience helper to create embeddings from chunks or strings.

    Args:
        chunks: List of chunk dictionaries containing 'text', or list of strings.
        model_name: Model identifier for SentenceTransformer.

    Returns:
        2D numpy array of float32 embeddings.
    """
    model = EmbeddingModel(model_name=model_name)
    if isinstance(chunks, (str, dict)):
        chunks = [chunks]
    texts = [
        c["text"] if isinstance(c, dict) and "text" in c else str(c)
        for c in chunks
    ]
    return model.encode(texts)