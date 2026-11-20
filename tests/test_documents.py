import json
import tempfile
import unittest
from pathlib import Path

import docx
from reportlab.pdfgen import canvas

from documents.chunker import TextChunker, chunk_document, chunk_documents
from documents.loader import DocumentLoader
from security.policy import Policy


def create_test_pdf(path: Path, page_texts: list[str]) -> None:
    c = canvas.Canvas(str(path))
    for text in page_texts:
        c.drawString(100, 700, text)
        c.showPage()
    c.save()


def create_test_docx(path: Path, paragraph_texts: list[str]) -> None:
    doc = docx.Document()
    for text in paragraph_texts:
        doc.add_paragraph(text)
    doc.save(str(path))


class TestDocumentIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        self.allowed_dir = self.base_path / "allowed"
        self.allowed_dir.mkdir()

        self.outside_dir = self.base_path / "outside"
        self.outside_dir.mkdir()

        # Config file setting allowed location
        self.config_file = self.base_path / "permissions.json"
        config_data = {"allowed_locations": [str(self.allowed_dir)]}
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        self.policy = Policy(config_path=self.config_file)
        self.loader = DocumentLoader(policy=self.policy)

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Loader Tests ---

    def test_loading_allowed_txt_file(self):
        txt_path = self.allowed_dir / "sample.txt"
        txt_path.write_text("Plain text content", encoding="utf-8")

        records = self.loader.load_file(txt_path)
        self.assertEqual(len(records), 1)
        self.assertIn("Plain text content", records[0]["text"])
        self.assertEqual(records[0]["source"], str(txt_path))
        self.assertIsNone(records[0]["page"])

    def test_loading_allowed_markdown_file(self):
        md_path = self.allowed_dir / "notes.md"
        md_path.write_text("# Section 1\nMarkdown body", encoding="utf-8")

        records = self.loader.load_file(md_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "# Section 1\nMarkdown body")
        self.assertEqual(records[0]["source"], str(md_path))
        self.assertIsNone(records[0]["page"])

    def test_loading_allowed_pdf(self):
        pdf_path = self.allowed_dir / "document.pdf"
        create_test_pdf(pdf_path, [
            "Page 1 content is long enough to exceed the ocr threshold.",
            "Page 2 content is long enough to exceed the ocr threshold.",
        ])

        records = self.loader.load_file(pdf_path)
        self.assertEqual(len(records), 2)

        self.assertIn("Page 1 content", records[0]["text"])
        self.assertEqual(records[0]["page"], 1)
        self.assertEqual(records[0]["source"], str(pdf_path))

        self.assertIn("Page 2 content", records[1]["text"])
        self.assertEqual(records[1]["page"], 2)
        self.assertEqual(records[1]["source"], str(pdf_path))

    def test_loading_allowed_docx(self):
        docx_path = self.allowed_dir / "document.docx"
        create_test_docx(docx_path, ["First paragraph.", "Second paragraph."])

        records = self.loader.load_file(docx_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "First paragraph.\nSecond paragraph.")
        self.assertEqual(records[0]["source"], str(docx_path))
        self.assertIsNone(records[0]["page"])

    def test_rejecting_unsupported_extension(self):
        py_file = self.allowed_dir / "script.py"
        py_file.write_text("print('hello')")

        with self.assertRaises(ValueError) as ctx:
            self.loader.load_file(py_file)
        self.assertIn("Unsupported document extension", str(ctx.exception))

    def test_rejecting_file_outside_permitted_directory(self):
        outside_file = self.outside_dir / "secret.txt"
        outside_file.write_text("secret data")

        with self.assertRaises(PermissionError):
            self.loader.load_file(outside_file)

    def test_preserving_pdf_page_numbers(self):
        pdf_path = self.allowed_dir / "multi_page.pdf"
        pages = [
            "Alpha page with sufficient text to pass the ocr threshold check.",
            "Beta page with sufficient text to pass the ocr threshold check.",
            "Gamma page with sufficient text to pass the ocr threshold check.",
        ]
        create_test_pdf(pdf_path, pages)

        records = self.loader.load_file(pdf_path)
        self.assertEqual(len(records), 3)
        page_nums = [r["page"] for r in records]
        self.assertEqual(page_nums, [1, 2, 3])

    def test_preserving_source_path(self):
        txt_path = self.allowed_dir / "source_test.txt"
        txt_path.write_text("test")

        records = self.loader.load_file(txt_path)
        self.assertEqual(records[0]["source"], str(txt_path.resolve()))

    # --- Chunker Semantic Tests ---

    def test_words_never_unnecessarily_split(self):
        doc = {
            "text": "SentinelAI is a privacy-first offline AI agent for secure documents.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=35, overlap=10)
        chunk_words = [w.strip(".,!?") for c in chunks for w in c["text"].split()]
        self.assertNotIn("vacy-first", chunk_words)
        self.assertIn("privacy-first", chunk_words)

    def test_sentences_remain_intact_when_possible(self):
        doc = {
            "text": "First sentence is complete. Second sentence is complete. Third sentence is complete.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=60, overlap=20)
        for chunk in chunks:
            # Check that every sentence inside chunk ends cleanly or is full sentence
            self.assertTrue(any(chunk["text"].endswith(end) for end in [".", "complete."]))

    def test_paragraph_boundaries_respected(self):
        doc = {
            "text": "Paragraph one content here.\n\nParagraph two content here.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=35, overlap=10)
        self.assertEqual(chunks[0]["text"], "Paragraph one content here.")
        self.assertEqual(chunks[1]["text"], "Paragraph two content here.")

    def test_overlap_contains_complete_semantic_units(self):
        doc = {
            "text": "Sentence A is here. Sentence B is here. Sentence C is here.",
            "source": "/doc.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=42, overlap=22)
        # Chunk 1 should have Sentence A & Sentence B
        # Chunk 2 overlap should include complete Sentence B
        self.assertIn("Sentence B is here.", chunks[0]["text"])
        self.assertTrue(chunks[1]["text"].startswith("Sentence B is here."))

    def test_source_and_page_metadata_preserved(self):
        doc = {
            "text": "Valid content for metadata preservation test.",
            "source": "/home/user/test.pdf",
            "page": 5
        }
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        for chunk in chunks:
            self.assertEqual(chunk["source"], "/home/user/test.pdf")
            self.assertEqual(chunk["page"], 5)

    def test_different_pdf_pages_remain_separate(self):
        page1_doc = {
            "text": "Page one text content.",
            "source": "/doc.pdf",
            "page": 1
        }
        page2_doc = {
            "text": "Page two text content.",
            "source": "/doc.pdf",
            "page": 2
        }
        chunks = chunk_documents([page1_doc, page2_doc], chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["text"], "Page one text content.")
        self.assertEqual(chunks[1]["page"], 2)
        self.assertEqual(chunks[1]["text"], "Page two text content.")

    def test_short_document_creates_one_chunk(self):
        doc = {
            "text": "Short document.",
            "source": "/short.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=500, overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Short document.")

    def test_empty_document_creates_no_chunks(self):
        doc = {
            "text": "",
            "source": "/empty.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=500, overlap=100)
        self.assertEqual(chunks, [])

    def test_whitespace_only_document_creates_no_chunks(self):
        doc = {
            "text": "   \n\t  \n  ",
            "source": "/blank.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=500, overlap=100)
        self.assertEqual(chunks, [])

    def test_long_sentence_handling(self):
        long_sentence = (
            "This is an exceptionally long sentence that easily exceeds the configured "
            "maximum target chunk size limit for testing word boundary splits."
        )
        doc = {
            "text": long_sentence,
            "source": "/long.txt",
            "page": None
        }
        chunks = chunk_document(doc, chunk_size=40, overlap=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Ensure no individual word was sliced in half across chunks
            words = chunk["text"].split()
            for w in words:
                self.assertIn(w, long_sentence)

    def test_configurable_chunk_size_and_overlap(self):
        doc = {
            "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa.",
            "source": "/config.txt",
            "page": None
        }
        chunker_large = TextChunker(chunk_size=100, overlap=20)
        chunks_large = chunker_large.chunk_document(doc)
        self.assertEqual(len(chunks_large), 1)

        chunker_small = TextChunker(chunk_size=25, overlap=5)
        chunks_small = chunker_small.chunk_document(doc)
        self.assertGreater(len(chunks_small), 1)


if __name__ == "__main__":
    unittest.main()
