from pathlib import Path
from typing import Any, Optional
import ollama

from documents.chunker import chunk_documents
from documents.loader import DocumentLoader, SUPPORTED_DOCUMENT_EXTENSIONS
from rag.generator import RAGGenerator, DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_BASE_URL
from rag.retriever import Retriever
from security.policy import Policy


class KnowledgeBase:
    """Coordinates automatic document discovery, loading, chunking, indexing, and RAG question answering."""

    def __init__(
        self,
        policy: Optional[Policy] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[RAGGenerator] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        client: Optional[ollama.Client] = None,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> None:
        """Initialize KnowledgeBase.

        Args:
            policy: Security Policy instance. Defaults to Policy().
            retriever: Retriever instance. Created automatically if None.
            generator: RAGGenerator instance. Created automatically if None.
            model_name: Ollama model name (default: qwen3:4b).
            ollama_base_url: Ollama endpoint URL (default: http://localhost:11434).
            client: Optional pre-configured ollama.Client for testing/mocking.
            chunk_size: Target character length per chunk.
            overlap: Target character overlap between chunks.
        """
        self.policy = policy if policy is not None else Policy()
        self.loader = DocumentLoader(policy=self.policy)
        self.chunk_size = chunk_size
        self.overlap = overlap

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = Retriever(policy=self.policy)

        if generator is not None:
            self.generator = generator
        else:
            self.generator = RAGGenerator(
                retriever=self.retriever,
                model_name=model_name,
                ollama_base_url=ollama_base_url,
                client=client
            )

        self.indexed_files: list[str] = []

    def discover_documents(self) -> list[Path]:
        """Discovers all supported document files recursively within policy-authorized locations.

        Enforces security Policy pre-checks on every file. Skips unreadable or unauthorized files safely.

        Returns:
            List of authorized Path objects.
        """
        discovered: list[Path] = []
        allowed_locations = self.policy.get_allowed_locations()

        for loc_str in allowed_locations:
            root_path = Path(loc_str).expanduser().resolve()
            if not root_path.exists():
                continue

            if root_path.is_file():
                if root_path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
                    if self.policy.is_allowed(root_path):
                        discovered.append(root_path)
                continue

            if not root_path.is_dir():
                continue

            # Recursively walk authorized root_path
            try:
                for path in root_path.rglob("*"):
                    try:
                        resolved_path = path.resolve()
                    except (OSError, RuntimeError):
                        continue

                    if not resolved_path.is_file():
                        continue

                    if resolved_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                        continue

                    # Security pre-check: Must be inside authorized directory
                    if self.policy.is_allowed(resolved_path):
                        discovered.append(resolved_path)
            except OSError:
                continue

        # Remove duplicates preserving discovery order
        unique_discovered: list[Path] = []
        seen = set()
        for p in discovered:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_discovered.append(p)

        return unique_discovered

    def build_index(self) -> int:
        """Discovers, loads, chunks, and indexes all authorized documents.

        Resets existing vector store index to prevent accumulating duplicate chunks.

        Returns:
            Total number of indexed chunks.
        """
        # Clear existing index to avoid duplicates
        self.retriever.reset()
        self.indexed_files = []

        files = self.discover_documents()
        all_chunks: list[dict[str, Any]] = []

        for file_path in files:
            try:
                # Load document records safely
                doc_records = self.loader.load_file(file_path)
                if not doc_records:
                    continue

                # Chunk loaded document records
                chunks = chunk_documents(
                    doc_records,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap
                )
                all_chunks.extend(chunks)
                self.indexed_files.append(str(file_path))
            except Exception:
                # Safely skip unreadable, malformed, or erroring files without crashing
                continue

        if not all_chunks:
            return 0

        indexed_count = self.retriever.index_documents(all_chunks)
        return indexed_count

    def refresh(self) -> int:
        """Refreshes knowledge base by rebuilding document index from scratch."""
        return self.build_index()

    def ask(
        self,
        question: str,
        k: int = 5,
        threshold: float = 0.35
    ) -> dict[str, Any]:
        """Answers user question using auto-indexed knowledge base without requiring a file path.

        Args:
            question: User question string.
            k: Maximum number of evidence chunks to retrieve.
            threshold: Similarity cutoff threshold.

        Returns:
            Dictionary containing 'answer', 'sources', and 'evidence_count'.
        """
        return self.generator.ask(question, k=k, threshold=threshold)
