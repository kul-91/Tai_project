import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from rag.embeddings import EmbeddingModel
from rag.knowledge_base import KnowledgeBase
from security.policy import Policy


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared EmbeddingModel to speed up test execution
        cls.shared_embedding_model = EmbeddingModel()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()

        self.nested_dir = self.allowed_dir / "nested"
        self.nested_dir.mkdir()

        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        self.config_file = self.base_path / "permissions.json"
        config_data = {"allowed_locations": [str(self.allowed_dir)]}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        self.policy = Policy(config_path=self.config_file)
        self.mock_client = MagicMock()

        self.kb = KnowledgeBase(
            policy=self.policy,
            client=self.mock_client
        )
        # Override embedding model with shared instance to avoid re-loading weights
        self.kb.retriever.embedding_model = self.shared_embedding_model
        self.kb.generator.retriever.embedding_model = self.shared_embedding_model

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discovering_files_recursively_in_authorized_directories(self):
        f1 = self.allowed_dir / "doc1.txt"
        f1.write_text("Document 1 content")

        f2 = self.nested_dir / "doc2.md"
        f2.write_text("# Section\nDocument 2 content")

        f_outside = self.outside_dir / "secret.txt"
        f_outside.write_text("Secret content")

        f_unsupported = self.allowed_dir / "script.py"
        f_unsupported.write_text("print('hello')")

        discovered = self.kb.discover_documents()
        discovered_paths = {p.resolve() for p in discovered}

        self.assertIn(f1.resolve(), discovered_paths)
        self.assertIn(f2.resolve(), discovered_paths)
        self.assertNotIn(f_outside.resolve(), discovered_paths)
        self.assertNotIn(f_unsupported.resolve(), discovered_paths)

    def test_indexing_multiple_files_and_answering_without_file_path(self):
        f1 = self.allowed_dir / "privacy.txt"
        f1.write_text("SentinelAI enforces strict privacy boundaries for local documents.")

        f2 = self.nested_dir / "recovery.md"
        f2.write_text("Emergency recovery code for system access is 9901.")

        indexed_chunks = self.kb.build_index()
        self.assertGreater(indexed_chunks, 0)

        self.mock_client.chat.return_value = {
            "message": {
                "content": "The emergency recovery code is 9901 [" + str(f2.resolve()) + "]."
            }
        }

        response = self.kb.ask("What is the emergency recovery code?")

        self.assertGreaterEqual(response["evidence_count"], 1)
        self.assertIn("9901", response["answer"])
        self.assertEqual(response["sources"][0]["source"], str(f2.resolve()))
        self.mock_client.chat.assert_called_once()

    def test_excluding_files_outside_authorized_directories(self):
        f_outside = self.outside_dir / "forbidden.txt"
        f_outside.write_text("Forbidden content")

        self.kb.build_index()
        results = self.kb.retriever.search("Forbidden content", k=5, threshold=0.1)

        for res in results:
            self.assertNotEqual(res["source"], str(f_outside.resolve()))

    def test_skipping_unsupported_and_unreadable_files(self):
        f_good = self.allowed_dir / "valid.txt"
        f_good.write_text("Valid text content for indexing.")

        f_bad_ext = self.allowed_dir / "data.bin"
        f_bad_ext.write_bytes(b"\x00\x01\x02\x03")

        indexed_count = self.kb.build_index()
        self.assertGreater(indexed_count, 0)
        self.assertIn(str(f_good.resolve()), self.kb.indexed_files)
        self.assertNotIn(str(f_bad_ext.resolve()), self.kb.indexed_files)

    def test_refreshing_after_file_changes_and_avoiding_duplicates(self):
        f1 = self.allowed_dir / "doc1.txt"
        f1.write_text("Initial document content for testing duplicate prevention.")

        count1 = self.kb.build_index()
        count2 = self.kb.build_index()

        # Rebuilding index must reset vector store and yield identical count (no duplicates)
        self.assertEqual(count1, count2)

        # Add new file and refresh
        f2 = self.allowed_dir / "doc2.txt"
        f2.write_text("Second document content added later.")

        count_refreshed = self.kb.refresh()
        self.assertGreater(count_refreshed, count1)

    def test_insufficient_evidence_for_unrelated_question(self):
        f1 = self.allowed_dir / "math.txt"
        f1.write_text("Linear algebra and matrix multiplication notes.")

        self.kb.build_index()

        response = self.kb.ask("What is the recipe for baking sourdough bread?", threshold=0.99)

        self.assertEqual(response["evidence_count"], 0)
        self.assertEqual(response["sources"], [])
        self.assertIn("do not contain enough information", response["answer"])
        self.mock_client.chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
