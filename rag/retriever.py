import re
from math import log
from typing import Any, Optional

import numpy as np

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from security.policy import Policy

# ─── Lexical search configuration ──────────────────────────────────────────

# Common English stop-words that should not dominate lexical scoring
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'of', 'to', 'for', 'and',
    'or', 'my', 'your', 'what', 'who', 'how', 'when', 'where', 'that',
    'this', 'with', 'by', 'at', 'from', 'be', 'was', 'are', 'were',
    'i', 'me', 'we', 'you', 'he', 'she', 'they', 'do', 'does', 'did',
    'has', 'have', 'had', 'not', 'no', 'so', 'but', 'as', 'if',
    'been', 'being', 'its', 'into', 'than', 'then', 'about',
})

_TOKEN_SPLIT = re.compile(r'[\s\u2014\u2013,;:.!?()\[\]/\u2022\u2023]+')


def _tokenize(text: str) -> list[str]:
    """Splits text into lowercase tokens, strips empty strings and stop-words."""
    tokens = []
    for raw in _TOKEN_SPLIT.split(text.lower()):
        t = raw.strip("-'\"")
        if t and t not in _STOP_WORDS and len(t) > 1:
            tokens.append(t)
    return tokens


def _stem(word: str) -> str:
    """Lightweight suffix stripper for English (not a full morphological stemmer)."""
    for suffix in ('ing', 'tion', 'ations', 'ation', 'ness', 'ment', 'ful', 'less', 'ly', 'ed', 'er', 'es', 's'):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _stem_tokens(tokens: list[str]) -> list[str]:
    return [_stem(t) for t in tokens]


class LexicalIndex:
    """Simple in-memory BM25-style lexical index over chunk texts."""

    # BM25 hyperparameters
    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._corpus: list[list[str]] = []      # stemmed token lists per chunk
        self._df: dict[str, int] = {}           # document frequency per term
        self._avg_dl: float = 0.0

    def add(self, texts: list[str]) -> None:
        """Indexes a list of chunk texts."""
        for text in texts:
            tokens = _stem_tokens(_tokenize(text))
            self._corpus.append(tokens)

        # Rebuild document frequency map
        self._df = {}
        for doc_tokens in self._corpus:
            for term in set(doc_tokens):
                self._df[term] = self._df.get(term, 0) + 1

        total_len = sum(len(doc) for doc in self._corpus)
        self._avg_dl = total_len / max(len(self._corpus), 1)

    def score(self, query: str) -> list[float]:
        """Returns a BM25 score for each indexed chunk against the query."""
        if not self._corpus:
            return []

        q_tokens = _stem_tokens(_tokenize(query))
        n_docs = len(self._corpus)
        scores: list[float] = [0.0] * n_docs

        for term in q_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, doc_tokens in enumerate(self._corpus):
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                dl = len(doc_tokens)
                tf_norm = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / max(self._avg_dl, 1))
                )
                scores[doc_idx] += idf * tf_norm

        return scores

    def reset(self) -> None:
        self._corpus = []
        self._df = {}
        self._avg_dl = 0.0


class Retriever:
    """Hybrid RAG Retriever combining dense semantic search (FAISS) with sparse BM25 lexical search.

    Retrieval flow:
        1. FAISS inner-product search on normalized embeddings → semantic scores (cosine similarity)
        2. BM25 scoring over all indexed chunk texts → lexical scores
        3. Both score types are min-max normalized independently to [0, 1]
        4. Combined score = semantic_weight * semantic_score + lexical_weight * lexical_score
        5. Results are ranked by combined score and filtered by hybrid threshold

    API compatibility:
        search(query, k, threshold) is preserved.
        'threshold' is now applied to the **combined** hybrid score (range [0, 1]).
        The default hybrid threshold (0.2) is lower than the old pure-cosine threshold (0.35)
        because the combined score reflects complementary signal; a chunk may have a high
        lexical score but moderate semantic score (or vice versa) and still be relevant.
        Callers that previously used threshold=0.35 on pure cosine should migrate to
        threshold=0.2 on the combined score, or pass threshold=0.35 explicitly for stricter
        filtering. The search() docstring documents the change.
    """

    # Default weights for hybrid combination
    DEFAULT_SEMANTIC_WEIGHT = 0.6
    DEFAULT_LEXICAL_WEIGHT = 0.4

    def __init__(
        self,
        policy: Policy,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ) -> None:
        """Initialize Retriever with Policy, EmbeddingModel, VectorStore, and hybrid weights.

        Args:
            policy: Security Policy instance enforcing access permissions.
            embedding_model: Optional EmbeddingModel instance.
            vector_store: Optional VectorStore instance.
            semantic_weight: Weight for normalized semantic score (default 0.6).
            lexical_weight: Weight for normalized lexical BM25 score (default 0.4).
        """
        if policy is None:
            raise ValueError('Retriever requires a valid Policy instance.')

        self.policy = policy
        self.embedding_model = embedding_model if embedding_model is not None else EmbeddingModel()
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.lexical_index = LexicalIndex()
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

        # Parallel metadata list matching VectorStore.metadata order
        self._metadata: list[dict[str, Any]] = []

    def index_documents(self, chunks: list[dict[str, Any]]) -> int:
        """Indexes document chunks into both FAISS and BM25 lexical index.

        Args:
            chunks: List of chunk dictionaries with 'text', 'source', and 'page'.

        Returns:
            Number of chunks successfully indexed.

        Raises:
            PermissionError: If any chunk's source path is denied by Policy.
        """
        if not chunks:
            return 0

        valid_chunks: list[dict[str, Any]] = []
        texts_to_embed: list[str] = []

        for chunk in chunks:
            source = chunk.get('source', '')
            # MANDATORY SECURITY CHECK: Verify source against Policy before any indexing
            self.policy.check_access(source)

            text = chunk.get('text', '')
            if text and text.strip():
                valid_chunks.append({
                    'text': text,
                    'source': source,
                    'page': chunk.get('page', None)
                })
                texts_to_embed.append(text)

        if not valid_chunks:
            return 0

        embeddings = self.embedding_model.encode(texts_to_embed)
        self.vector_store.add(embeddings, valid_chunks)
        self.lexical_index.add(texts_to_embed)
        self._metadata.extend(valid_chunks)

        return len(valid_chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining dense semantic and sparse BM25 lexical relevance.

        The 'threshold' parameter now applies to the **combined hybrid score** (range [0, 1]).
        A value of 0.2 is recommended as the default because:
        - Chunks with strong lexical overlap but moderate semantic similarity should surface.
        - The semantic weight (0.6) still ensures semantic irrelevance suppresses low-quality matches.

        API change from pure-cosine retriever:
        - Old: threshold applied to raw cosine similarity (FAISS inner product, range [-1, 1]).
        - New: threshold applied to weighted combination of normalized semantic + BM25 scores.
        - Callers relying on threshold=0.35 for strict semantic filtering should pass
          semantic_threshold separately or adjust to the new combined-score scale.

        Args:
            query: Query string.
            k: Maximum number of results to return.
            threshold: Minimum combined hybrid score cutoff (range [0, 1], default 0.2).

        Returns:
            List of matching records sorted descending by combined score:
            [
                {
                    "text": str,
                    "source": str,
                    "page": Optional[int],
                    "score": float,         # combined hybrid score
                    "semantic_score": float, # normalized cosine similarity
                    "lexical_score": float   # normalized BM25 score
                }
            ]
        """
        if not query or not query.strip():
            return []

        n_total = self.vector_store.index.ntotal
        if n_total == 0:
            return []

        # ── Step 1: Dense semantic retrieval (broad k for ranking fusion) ──────
        query_vector = self.embedding_model.encode_query(query)
        if query_vector.size == 0:
            return []

        # Retrieve up to n_total candidates for proper hybrid ranking
        broad_k = min(max(k * 4, 20), n_total)

        if query_vector.ndim == 1:
            qv = np.expand_dims(query_vector, axis=0).astype(np.float32)
        else:
            qv = query_vector.astype(np.float32)

        raw_scores, raw_indices = self.vector_store.index.search(qv, broad_k)
        sem_scores_raw = raw_scores[0]           # shape: (broad_k,)
        sem_indices = raw_indices[0]             # shape: (broad_k,)

        # ── Step 2: BM25 lexical scoring over ALL indexed chunks ─────────────
        lex_scores_all = self.lexical_index.score(query)  # list[float], len == n_total

        # ── Step 3: Collect candidate set and their raw scores ───────────────
        # Candidates from semantic search (valid indices only)
        candidate_set: dict[int, dict[str, float]] = {}
        for raw_s, idx in zip(sem_scores_raw, sem_indices):
            if idx < 0 or idx >= n_total:
                continue
            if idx not in candidate_set:
                candidate_set[idx] = {'sem_raw': float(raw_s), 'lex_raw': 0.0}
            if idx < len(lex_scores_all):
                candidate_set[idx]['lex_raw'] = lex_scores_all[idx]

        # Also add high lexical-scoring chunks that may have been missed by FAISS
        if lex_scores_all:
            lex_threshold = 0.0
            for chunk_idx, lex_s in enumerate(lex_scores_all):
                if lex_s > lex_threshold and chunk_idx not in candidate_set:
                    # Compute semantic score for this chunk on demand
                    # We can't re-search per-chunk in FAISS easily without
                    # re-encoding, so assign 0.0 semantic for lex-only additions
                    candidate_set[chunk_idx] = {'sem_raw': 0.0, 'lex_raw': lex_s}

        # ── Step 4: Normalize scores to [0, 1] independently ────────────────
        #
        # Semantic: FAISS IndexFlatIP returns inner products which, for unit-norm
        # vectors, equal cosine similarity in [−1, 1].  We clip to [0, 1] using
        # an absolute scale rather than min-max-normalizing over the candidate
        # set.  Min-max normalization over a small pool (e.g. 2 chunks) would
        # always assign one chunk score=1.0 regardless of how poor its actual
        # cosine similarity is, masking the lexical signal.
        #
        # Lexical (BM25): Min-max normalize over the candidate set so that the
        # best keyword match always contributes its full lexical_weight.

        lex_raws = [v['lex_raw'] for v in candidate_set.values()]
        lex_min = min(lex_raws) if lex_raws else 0.0
        lex_max = max(lex_raws) if lex_raws else 1.0

        def norm_sem(v: float) -> float:
            """Clip raw cosine similarity to [0, 1]."""
            return max(0.0, min(1.0, float(v)))

        def norm_lex(v: float) -> float:
            """Min-max normalize BM25 score over the candidate pool."""
            if lex_max == lex_min:
                return 0.0
            return (v - lex_min) / (lex_max - lex_min)

        # ── Step 5: Compute combined scores and apply threshold ───────────────
        results: list[dict[str, Any]] = []
        for idx, raw in candidate_set.items():
            if idx >= len(self._metadata):
                continue
            s_norm = norm_sem(raw['sem_raw'])
            l_norm = norm_lex(raw['lex_raw'])
            combined = self.semantic_weight * s_norm + self.lexical_weight * l_norm
            if combined >= threshold:
                record = dict(self._metadata[idx])
                record['score'] = round(combined, 4)
                record['semantic_score'] = round(s_norm, 4)
                record['lexical_score'] = round(l_norm, 4)
                results.append(record)

        # Sort descending by combined score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def reset(self) -> None:
        """Resets the vector store index, lexical index, and metadata."""
        self.vector_store.reset()
        self.lexical_index.reset()
        self._metadata = []
