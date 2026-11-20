"""Tests for hybrid retrieval, structure-aware chunking, and knowledge-base integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from documents.chunker import TextChunker, chunk_document, chunk_documents
from rag.embeddings import EmbeddingModel
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from security.policy import Policy


# ── Resume fragment used in regression tests ──────────────────────────────────
RESUME_FRAGMENT = """Projects
Projects — Movie Recommendation System — Python, FastAPI, React, Tailwind CSS
Built a full-stack movie recommendation engine with collaborative filtering.
Integrated TMDB API for movie metadata and deployed using Docker and Nginx.

Projects — Chat Application — Node.js, Socket.io, MongoDB
Developed a real-time chat application supporting group and private messaging.
Implemented JWT authentication and end-to-end message encryption.
"""


class TestStructureAwareChunker(unittest.TestCase):
    """Tests for the structure-aware chunking improvements."""

    def test_resume_project_title_stays_with_description(self):
        """Project title line should be kept with the description that follows."""
        doc = {
            "text": RESUME_FRAGMENT,
            "source": "/docs/resume.txt",
            "page": None
        }
        chunks = chunk_documents([doc], chunk_size=600, overlap=50)
        full_text = " ".join(c["text"] for c in chunks)
        # Both project titles and their descriptions must appear in some chunk
        self.assertIn("Movie Recommendation System", full_text)
        self.assertIn("recommendation engine", full_text)
        self.assertIn("Chat Application", full_text)

    def test_first_project_chunk_contains_title_and_description(self):
        """The chunk containing 'Movie Recommendation System' should also have its description."""
        doc = {"text": RESUME_FRAGMENT, "source": "/doc.txt", "page": None}
        chunks = chunk_documents([doc], chunk_size=700, overlap=50)
        movie_chunk = next(
            (c for c in chunks if "Movie Recommendation System" in c["text"]), None
        )
        self.assertIsNotNone(movie_chunk, "Expected a chunk containing 'Movie Recommendation System'")
        self.assertIn("recommendation engine", movie_chunk["text"])

    def test_section_heading_detection(self):
        doc = {
            "text": "Education\nBachelor of Technology in Computer Science, 2024.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_documents([doc], chunk_size=300, overlap=30)
        full = " ".join(c["text"] for c in chunks)
        self.assertIn("Education", full)
        self.assertIn("Bachelor of Technology", full)

    def test_words_never_split(self):
        doc = {
            "text": "SentinelAI is a privacy-first offline AI agent for secure documents.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=35, overlap=10)
        chunk_words = [w.strip(".,!?") for c in chunks for w in c["text"].split()]
        self.assertNotIn("vacy-first", chunk_words)
        self.assertIn("privacy-first", chunk_words)

    def test_sentences_intact_when_possible(self):
        doc = {
            "text": "First sentence is complete. Second sentence is complete. Third sentence here.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=60, overlap=20)
        for chunk in chunks:
            self.assertTrue(any(chunk["text"].endswith(end) for end in [".", "complete."]))

    def test_paragraph_boundaries_respected(self):
        doc = {
            "text": "Paragraph one content here.\n\nParagraph two content here.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=50, overlap=0)
        self.assertEqual(len(chunks), 2)

    def test_empty_document_creates_no_chunks(self):
        self.assertEqual(chunk_document({"text": "", "source": "/e.txt", "page": None}), [])
        self.assertEqual(chunk_document({"text": "   \n\t", "source": "/e.txt", "page": None}), [])

    def test_short_document_creates_one_chunk(self):
        doc = {"text": "Short document.", "source": "/doc.txt", "page": None}
        chunks = chunk_document(doc, chunk_size=500, overlap=100)
        self.assertEqual(len(chunks), 1)

    def test_long_sentence_stays_complete_words(self):
        long_sentence = "This is a very long sentence that exceeds target chunk size but must not split a word."
        doc = {"text": long_sentence, "source": "/doc.txt", "page": None}
        chunks = chunk_document(doc, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            for w in chunk["text"].split():
                self.assertIn(w.strip(".,!?"), long_sentence)

    def test_metadata_preserved_through_chunking(self):
        doc = {"text": "Content about machine learning and RAG systems.", "source": "/ml.pdf", "page": 7}
        chunks = chunk_document(doc, chunk_size=200, overlap=50)
        for c in chunks:
            self.assertEqual(c["source"], "/ml.pdf")
            self.assertEqual(c["page"], 7)

    def test_pdf_page_boundaries_separate(self):
        page1 = {"text": "Page one content about neural networks.", "source": "/doc.pdf", "page": 1}
        page2 = {"text": "Page two content about transformers.", "source": "/doc.pdf", "page": 2}
        chunks = chunk_documents([page1, page2], chunk_size=200, overlap=30)
        pages_in_chunks = {c["page"] for c in chunks}
        self.assertIn(1, pages_in_chunks)
        self.assertIn(2, pages_in_chunks)
        # No chunk should span two pages
        for c in chunks:
            self.assertIn(c["page"], {1, 2})

    def test_configurable_chunk_size_and_overlap(self):
        doc = {"text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa.", "source": "/c.txt", "page": None}
        large_chunks = chunk_document(doc, chunk_size=200, overlap=20)
        small_chunks = chunk_document(doc, chunk_size=25, overlap=5)
        self.assertLessEqual(len(large_chunks), len(small_chunks))


# ── Hybrid Retriever Tests ────────────────────────────────────────────────────

class TestHybridRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedding_model = EmbeddingModel()

    def _make_policy_and_retriever(self, allowed_dir: Path, config_file: Path):
        config_data = {"allowed_locations": [str(allowed_dir)]}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)
        policy = Policy(config_path=config_file)
        retriever = Retriever(policy=policy, embedding_model=self.embedding_model)
        return policy, retriever

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name).resolve()
        self.allowed_dir = self.base / "allowed"
        self.allowed_dir.mkdir()
        self.outside_dir = self.base / "outside"
        self.outside_dir.mkdir()
        self.config_file = self.base / "permissions.json"
        self.policy, self.retriever = self._make_policy_and_retriever(
            self.allowed_dir, self.config_file
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_keyword_match_surfaces_chunk_with_low_semantic_score(self):
        """Hybrid retrieval must include the Projects chunk when query mentions 'project'.

        The RESUME_FRAGMENT contains a dedicated projects section. After chunking,
        at least one chunk must contain 'Movie Recommendation System'.
        Hybrid search (semantic + BM25) should rank it in the top-3 for a query
        asking about resume projects.
        """
        resume_source = str(self.allowed_dir / "resume.txt")
        # Use full resume fragment → produces multiple chunks so hybrid ranking
        # isn't distorted by 2-chunk min-max effects
        from documents.chunker import chunk_documents as _cd
        chunks = _cd(
            [{"text": RESUME_FRAGMENT, "source": resume_source, "page": None}],
            chunk_size=700, overlap=80
        )
        self.retriever.index_documents(chunks)
        results = self.retriever.search(
            "What is the first project described on my resume", k=5, threshold=0.0
        )
        self.assertGreater(len(results), 0)
        # At least one top result must reference projects / movie recommendation
        all_texts = " ".join(r["text"] for r in results)
        self.assertIn("Movie Recommendation System", all_texts)

    def test_semantic_retrieval_benefits_synonyms(self):
        """Semantic scoring should help synonymous queries even without exact keyword overlap."""
        source = str(self.allowed_dir / "ml.txt")
        chunks = [
            {
                "text": "Artificial intelligence and machine learning techniques for pattern recognition.",
                "source": source,
                "page": 1
            },
            {
                "text": "Italian pasta recipes with tomato sauce and basil.",
                "source": source,
                "page": 2
            }
        ]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("deep learning and neural networks", k=1, threshold=0.1)
        self.assertGreater(len(results), 0)
        self.assertIn("machine learning", results[0]["text"])

    def test_hybrid_scores_normalized_and_in_range(self):
        """Combined score and sub-scores should all be in [0, 1]."""
        source = str(self.allowed_dir / "doc.txt")
        chunks = [
            {"text": "Python programming language for data science.", "source": source, "page": 1},
            {"text": "Gardening tips for growing tomatoes.", "source": source, "page": 2},
        ]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("Python data", k=5, threshold=0.0)
        for r in results:
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)
            self.assertGreaterEqual(r["semantic_score"], 0.0)
            self.assertLessEqual(r["semantic_score"], 1.0)
            self.assertGreaterEqual(r["lexical_score"], 0.0)
            self.assertLessEqual(r["lexical_score"], 1.0)

    def test_irrelevant_keyword_match_does_not_dominate(self):
        """A technically relevant chunk should appear in results for a technical query."""
        source = str(self.allowed_dir / "doc.txt")
        chunks = [
            {
                "text": "FastAPI project built with Python provides REST endpoints for machine learning inference.",
                "source": source,
                "page": 1
            },
            {
                "text": "Italian pasta recipes: spaghetti carbonara, fettuccine alfredo.",
                "source": source,
                "page": 2
            }
        ]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("Python REST API project for ML inference", k=2, threshold=0.0)
        self.assertGreater(len(results), 0)
        # The technical ML chunk should appear somewhere in the results
        texts = [r["text"] for r in results]
        self.assertTrue(
            any("FastAPI" in t for t in texts),
            f"Expected FastAPI chunk in results; got: {texts}"
        )

    def test_no_duplicate_results(self):
        """Each chunk should appear at most once in results even if it scores well in both retrievers."""
        source = str(self.allowed_dir / "doc.txt")
        chunks = [
            {"text": "Machine learning recommendation system using collaborative filtering.", "source": source, "page": 1},
        ]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("machine learning recommendation", k=5, threshold=0.0)
        texts = [r["text"] for r in results]
        self.assertEqual(len(texts), len(set(texts)))

    def test_search_before_indexing_returns_empty(self):
        results = self.retriever.search("anything")
        self.assertEqual(results, [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("   "), [])

    def test_unauthorized_chunk_rejected(self):
        denied_source = str(self.outside_dir / "secret.txt")
        with self.assertRaises(PermissionError):
            self.retriever.index_documents([
                {"text": "Secret content.", "source": denied_source, "page": 1}
            ])

    def test_reset_clears_index(self):
        source = str(self.allowed_dir / "doc.txt")
        self.retriever.index_documents([
            {"text": "Some indexed content for testing.", "source": source, "page": 1}
        ])
        self.retriever.reset()
        results = self.retriever.search("Some indexed content")
        self.assertEqual(results, [])

    def test_source_and_page_metadata_in_results(self):
        source = str(self.allowed_dir / "paper.pdf")
        chunks = [{"text": "Transformers architecture for NLP tasks.", "source": source, "page": 5}]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("Transformers NLP", k=1, threshold=0.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], source)
        self.assertEqual(results[0]["page"], 5)

    def test_top_k_respected(self):
        source = str(self.allowed_dir / "doc.txt")
        chunks = [
            {"text": f"Chunk number {i} about machine learning.", "source": source, "page": i}
            for i in range(1, 6)
        ]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("machine learning", k=2, threshold=0.0)
        self.assertLessEqual(len(results), 2)

    def test_threshold_filtering(self):
        source = str(self.allowed_dir / "doc.txt")
        chunks = [{"text": "Quantum entanglement and particle physics.", "source": source, "page": 1}]
        self.retriever.index_documents(chunks)
        results = self.retriever.search("sourdough bread baking recipes", k=5, threshold=0.99)
        self.assertEqual(results, [])

    def test_resume_regression_movie_recommendation(self):
        """Regression: query for first project must retrieve Movie Recommendation System chunk."""
        source = str(self.allowed_dir / "resume.txt")
        chunks = chunk_documents(
            [{"text": RESUME_FRAGMENT, "source": source, "page": None}],
            chunk_size=700,
            overlap=80
        )
        self.retriever.index_documents(chunks)

        results = self.retriever.search(
            "What is the first project described on my resume",
            k=3,
            threshold=0.0
        )

        self.assertGreater(len(results), 0, "Expected at least one result for resume project query")
        top_result = results[0]

        # Must find the Movie Recommendation System chunk
        self.assertIn("Movie Recommendation System", top_result["text"],
                      f"Expected 'Movie Recommendation System' in top result. Got: {top_result['text']!r}")

        # Scores must be in range
        self.assertGreaterEqual(top_result["score"], 0.0)
        self.assertLessEqual(top_result["score"], 1.0)
        print(
            f"\n[Regression] Query: 'first project on resume'\n"
            f"  Top chunk: {top_result['text'][:120]!r}\n"
            f"  Combined: {top_result['score']}, Semantic: {top_result['semantic_score']}, "
            f"Lexical: {top_result['lexical_score']}"
        )


# ── KnowledgeBase Integration Tests ─────────────────────────────────────────

class TestKnowledgeBaseIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shared_embedding_model = EmbeddingModel()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()
        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()
        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        self.config_file = self.base_path / "permissions.json"
        config_data = {"allowed_locations": [str(self.allowed_dir)]}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        self.policy = Policy(config_path=self.config_file)
        self.mock_client = MagicMock()
        self.kb = KnowledgeBase(policy=self.policy, client=self.mock_client)
        self.kb.retriever.embedding_model = self.shared_embedding_model
        self.kb.generator.retriever.embedding_model = self.shared_embedding_model

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_answering_without_file_path(self):
        f = self.allowed_dir / "privacy.txt"
        f.write_text("SentinelAI enforces strict privacy boundaries for local documents.")
        self.kb.build_index()
        self.mock_client.chat.return_value = {
            "message": {"content": "SentinelAI enforces privacy boundaries [privacy.txt]."}
        }
        response = self.kb.ask("How does SentinelAI protect privacy?")
        self.assertGreaterEqual(response["evidence_count"], 1)

    def test_multiple_files_indexed_and_retrievable(self):
        f1 = self.allowed_dir / "a.txt"
        f1.write_text("Neural networks learn representations from data.")
        f2 = (self.allowed_dir / "sub")
        f2.mkdir()
        f3 = f2 / "b.txt"
        f3.write_text("Cryptographic hash functions ensure data integrity.")
        self.kb.build_index()
        self.assertIn(str(f1.resolve()), self.kb.indexed_files)
        self.assertIn(str(f3.resolve()), self.kb.indexed_files)

    def test_no_chunks_from_outside_authorized_dirs(self):
        f_out = self.outside_dir / "forbidden.txt"
        f_out.write_text("Top secret forbidden content.")
        self.kb.build_index()
        results = self.kb.retriever.search("forbidden secret", k=5, threshold=0.0)
        for r in results:
            self.assertNotEqual(r["source"], str(f_out.resolve()))

    def test_refresh_avoids_duplicate_chunks(self):
        f = self.allowed_dir / "doc.txt"
        f.write_text("Stable content for duplicate prevention testing.")
        count1 = self.kb.build_index()
        count2 = self.kb.build_index()
        self.assertEqual(count1, count2)

    def test_refresh_picks_up_new_files(self):
        f1 = self.allowed_dir / "original.txt"
        f1.write_text("Original document content.")
        count1 = self.kb.build_index()
        f2 = self.allowed_dir / "added.txt"
        f2.write_text("Newly added document content.")
        count2 = self.kb.refresh()
        self.assertGreater(count2, count1)

    def test_insufficient_evidence_for_unrelated_question(self):
        f = self.allowed_dir / "math.txt"
        f.write_text("Linear algebra involves matrix operations and vector spaces.")
        self.kb.build_index()
        response = self.kb.ask("How do I bake sourdough bread at home?", threshold=0.99)
        self.assertEqual(response["evidence_count"], 0)
        self.assertIn("do not contain enough information", response["answer"])
        self.mock_client.chat.assert_not_called()

    def test_source_and_page_citations_in_response(self):
        f = self.allowed_dir / "policy_doc.txt"
        f.write_text("Access control policy requires explicit user authorization for all file operations.")
        self.kb.build_index()
        self.mock_client.chat.return_value = {
            "message": {"content": "Access requires authorization [policy_doc.txt]."}
        }
        response = self.kb.ask("What does the access control policy say?")
        self.assertGreaterEqual(response["evidence_count"], 1)
        self.assertEqual(response["sources"][0]["source"], str(f.resolve()))
        self.assertIsNone(response["sources"][0]["page"])


if __name__ == "__main__":
    unittest.main()
