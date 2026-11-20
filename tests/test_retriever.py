import json
import tempfile
import unittest
from pathlib import Path

from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from security.policy import Policy


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared EmbeddingModel across tests to save initialization time
        cls.shared_embedding_model = EmbeddingModel()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()

        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        # Config setting allowed location
        self.config_file = self.base_path / "permissions.json"
        config_data = {"allowed_locations": [str(self.allowed_dir)]}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        self.policy = Policy(config_path=self.config_file)
        self.retriever = Retriever(
            policy=self.policy,
            embedding_model=self.shared_embedding_model,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_indexing_permitted_chunks(self):
        chunks = [
            {
                "text": "SentinelAI ensures offline privacy for AI models.",
                "source": str(self.allowed_dir / "guide.txt"),
                "page": 1
            }
        ]
        indexed_count = self.retriever.index_documents(chunks)
        self.assertEqual(indexed_count, 1)

    def test_rejecting_chunks_outside_permitted_locations(self):
        denied_chunk = [
            {
                "text": "Confidential data stored outside.",
                "source": str(self.outside_dir / "secret.txt"),
                "page": 1
            }
        ]
        with self.assertRaises(PermissionError):
            self.retriever.index_documents(denied_chunk)

    def test_searching_relevant_content_and_metadata(self):
        chunks = [
            {
                "text": "Python is a popular programming language for machine learning.",
                "source": str(self.allowed_dir / "python_doc.txt"),
                "page": 3
            },
            {
                "text": "Baking recipes for chocolate cakes and desserts.",
                "source": str(self.allowed_dir / "recipes.txt"),
                "page": 1
            }
        ]
        self.retriever.index_documents(chunks)

        results = self.retriever.search("Python machine learning code", k=1, threshold=0.0)
        self.assertEqual(len(results), 1)
        self.assertIn("Python is a popular programming language", results[0]["text"])
        self.assertEqual(results[0]["source"], str(self.allowed_dir / "python_doc.txt"))
        self.assertEqual(results[0]["page"], 3)
        self.assertIn("score", results[0])
        # Hybrid retriever returns combined score, semantic_score, lexical_score
        self.assertIn("semantic_score", results[0])
        self.assertIn("lexical_score", results[0])
        self.assertGreaterEqual(results[0]["score"], 0.0)

    def test_top_k_behavior(self):
        chunks = [
            {"text": f"Information chunk number {i}", "source": str(self.allowed_dir / "doc.txt"), "page": i}
            for i in range(1, 6)
        ]
        self.retriever.index_documents(chunks)

        results = self.retriever.search("Information chunk", k=2, threshold=0.0)
        self.assertLessEqual(len(results), 2)

    def test_threshold_filtering(self):
        chunks = [
            {
                "text": "Quantum physics and particle accelerators in modern science.",
                "source": str(self.allowed_dir / "science.txt"),
                "page": 1
            }
        ]
        self.retriever.index_documents(chunks)

        # Very high threshold for unrelated query should return empty list
        results = self.retriever.search("Banana smoothie dessert recipe", k=5, threshold=0.95)
        self.assertEqual(results, [])

    def test_searching_before_indexing(self):
        results = self.retriever.search("Query before indexing")
        self.assertEqual(results, [])

    def test_multiple_documents_and_pages(self):
        chunks = [
            {
                "text": "Document 1 page 1 content regarding privacy.",
                "source": str(self.allowed_dir / "doc1.pdf"),
                "page": 1
            },
            {
                "text": "Document 1 page 2 content regarding encryption.",
                "source": str(self.allowed_dir / "doc1.pdf"),
                "page": 2
            },
            {
                "text": "Document 2 page 1 content regarding user permissions.",
                "source": str(self.allowed_dir / "doc2.pdf"),
                "page": 1
            }
        ]
        self.retriever.index_documents(chunks)

        results = self.retriever.search("encryption and privacy", k=3, threshold=0.1)
        self.assertGreaterEqual(len(results), 1)

        sources = {r["source"] for r in results}
        pages = {r["page"] for r in results}
        self.assertTrue(sources.issubset({str(self.allowed_dir / "doc1.pdf"), str(self.allowed_dir / "doc2.pdf")}))
        self.assertTrue(pages.issubset({1, 2}))

    def test_empty_input(self):
        self.assertEqual(self.retriever.index_documents([]), 0)
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("   "), [])


if __name__ == "__main__":
    unittest.main()
