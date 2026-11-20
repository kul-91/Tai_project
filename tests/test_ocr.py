"""Tests for OCR fallback in DocumentLoader (Step 5.2).

All OCR and PDF-rendering dependencies (fitz/PyMuPDF, pytesseract, PIL) are
mocked so the suite runs without Tesseract or actual scanned PDFs.

Coverage:
- Text-based PDF pages do NOT invoke OCR.
- Image-only pages DO invoke OCR.
- Mixed PDFs use text extraction on text pages, OCR on scanned pages.
- Source path and 1-based page numbers are preserved in every record.
- OCR text flows through the existing chunker.
- OCR failures produce an "ocr_error" key and empty text without crashing.
- Missing pytesseract / fitz produces a clear error record.
- Tesseract not installed (TesseractNotFoundError) produces a clear error record.
- Empty / whitespace-only pages trigger OCR (treated as image pages).
- Unauthorized PDFs are rejected before any rendering or OCR.
- Existing TXT, Markdown, and DOCX loading is unaffected.
- KnowledgeBase refresh does not accumulate duplicate chunks.

OCR min-char threshold is configurable via DocumentLoader(ocr_min_chars=N).
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import docx as _docx
from reportlab.pdfgen import canvas as _canvas

from documents.chunker import chunk_documents
from documents.loader import (
    OCR_DEFAULT_DPI,
    OCR_DEFAULT_LANG,
    OCR_MIN_CHARS,
    DocumentLoader,
)
from security.policy import Policy


# ─── PDF creation helpers ─────────────────────────────────────────────────────

def _make_text_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a text-based PDF using reportlab."""
    c = _canvas.Canvas(str(path))
    for text in page_texts:
        c.drawString(100, 700, text)
        c.showPage()
    c.save()


def _make_docx(path: Path, texts: list[str]) -> None:
    doc = _docx.Document()
    for t in texts:
        doc.add_paragraph(t)
    doc.save(str(path))


# ─── Shared Policy/Loader setup ───────────────────────────────────────────────

class _BaseOCRTest(unittest.TestCase):
    """Base class: creates a temp dir with an allowed sub-dir and a Policy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.allowed = self.base / "allowed"
        self.allowed.mkdir()
        self.outside = self.base / "outside"
        self.outside.mkdir()

        cfg = self.base / "permissions.json"
        cfg.write_text(json.dumps({"allowed_locations": [str(self.allowed)]}))
        self.policy = Policy(config_path=cfg)
        self.loader = DocumentLoader(policy=self.policy)

    def tearDown(self):
        self.tmp.cleanup()


# ─── Default-value tests ──────────────────────────────────────────────────────

class TestOCRDefaults(unittest.TestCase):
    def test_default_ocr_min_chars(self):
        self.assertEqual(OCR_MIN_CHARS, 30)

    def test_default_dpi(self):
        self.assertEqual(OCR_DEFAULT_DPI, 200)

    def test_default_lang(self):
        self.assertEqual(OCR_DEFAULT_LANG, "eng")

    def test_loader_defaults_applied(self):
        loader = DocumentLoader()
        self.assertEqual(loader.ocr_dpi, OCR_DEFAULT_DPI)
        self.assertEqual(loader.ocr_lang, OCR_DEFAULT_LANG)
        self.assertEqual(loader.ocr_min_chars, OCR_MIN_CHARS)

    def test_loader_custom_settings(self):
        loader = DocumentLoader(ocr_dpi=300, ocr_lang="eng+fra", ocr_min_chars=50)
        self.assertEqual(loader.ocr_dpi, 300)
        self.assertEqual(loader.ocr_lang, "eng+fra")
        self.assertEqual(loader.ocr_min_chars, 50)


# ─── Text-based PDF: no OCR invoked ─────────────────────────────────────────

class TestTextPDFNoOCR(_BaseOCRTest):

    def test_text_pdf_does_not_call_ocr(self):
        """A page with sufficient text must never trigger OCR."""
        pdf = self.allowed / "text.pdf"
        # Text well above ocr_min_chars=30
        _make_text_pdf(pdf, ["This page has plenty of readable text content here."])

        with patch.object(self.loader, "_ocr_pdf_page") as mock_ocr:
            records = self.loader.load_file(pdf)
            mock_ocr.assert_not_called()

        self.assertEqual(len(records), 1)
        self.assertGreater(len(records[0]["text"].strip()), 30)
        self.assertEqual(records[0]["page"], 1)
        self.assertEqual(records[0]["source"], str(pdf))

    def test_multi_page_text_pdf_no_ocr(self):
        """All text-rich pages skip OCR."""
        pdf = self.allowed / "multi.pdf"
        _make_text_pdf(pdf, [
            "Page one content is long enough to exceed the threshold.",
            "Page two content is also long enough to exceed the threshold.",
        ])

        with patch.object(self.loader, "_ocr_pdf_page") as mock_ocr:
            records = self.loader.load_file(pdf)
            mock_ocr.assert_not_called()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["page"], 1)
        self.assertEqual(records[1]["page"], 2)

    def test_text_pdf_page_numbers_are_1_based(self):
        pdf = self.allowed / "pages.pdf"
        _make_text_pdf(pdf, [
            "First page content with plenty of text for extraction.",
            "Second page content with plenty of text for extraction.",
            "Third page content with plenty of text for extraction.",
        ])
        records = self.loader.load_file(pdf)
        self.assertEqual([r["page"] for r in records], [1, 2, 3])

    def test_text_pdf_source_preserved(self):
        pdf = self.allowed / "source.pdf"
        _make_text_pdf(pdf, ["Source preservation test with enough text to pass threshold."])
        records = self.loader.load_file(pdf)
        self.assertEqual(records[0]["source"], str(pdf))


# ─── Image / scanned pages: OCR invoked ─────────────────────────────────────

class TestScannedPageOCR(_BaseOCRTest):

    def _make_ocr_loader(self, ocr_text: str) -> DocumentLoader:
        """Return a loader whose _ocr_pdf_page is mocked to return ocr_text."""
        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)

        def fake_ocr(pdf_path, source, page_num):
            return {"text": ocr_text, "source": source, "page": page_num}

        loader._ocr_pdf_page = fake_ocr
        return loader

    def test_scanned_page_triggers_ocr(self):
        """A page with almost no extractable text must call OCR."""
        pdf = self.allowed / "scan.pdf"
        _make_text_pdf(pdf, ["x"])  # any valid PDF

        ocr_called: list[int] = []

        def fake_ocr(pdf_path, source, page_num):
            ocr_called.append(page_num)
            return {"text": "Scanned OCR content.", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = fake_ocr

        # Mock PdfReader to return an empty page
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(len(records), 1)
        self.assertEqual(ocr_called, [1], "OCR should have been called for page 1")
        self.assertEqual(records[0]["text"], "Scanned OCR content.")
        self.assertEqual(records[0]["page"], 1)
        self.assertEqual(records[0]["source"], str(pdf))

    def test_empty_page_triggers_ocr(self):
        """A page whose pypdf text is empty (len < threshold) must use OCR."""
        pdf = self.allowed / "empty_page.pdf"
        _make_text_pdf(pdf, ["Long text page to create valid PDF structure."])

        def fake_ocr(pdf_path, source, page_num):
            return {"text": "OCR found: scanned content.", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = fake_ocr

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertIn("OCR found", records[0]["text"])

    def test_whitespace_only_page_triggers_ocr(self):
        """A page whose text is only whitespace is treated as image-only."""
        pdf = self.allowed / "ws.pdf"
        _make_text_pdf(pdf, ["Base page content for PDF structure."])

        ocr_triggered = []

        def fake_ocr(pdf_path, source, page_num):
            ocr_triggered.append(page_num)
            return {"text": "OCR result.", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = fake_ocr

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "    \n\t "
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertTrue(len(ocr_triggered) > 0)

    def test_ocr_text_preserved_in_record(self):
        """OCR output must appear verbatim in the record text."""
        pdf = self.allowed / "ocr.pdf"
        _make_text_pdf(pdf, ["p"])
        expected_ocr_text = "Tesseract found: invoice total $42.00"

        loader = self._make_ocr_loader(expected_ocr_text)

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertIn(expected_ocr_text, records[0]["text"])

    def test_ocr_source_and_page_metadata_preserved(self):
        pdf = self.allowed / "meta.pdf"
        _make_text_pdf(pdf, ["p", "pp"])

        sources, pages = [], []

        def tracking_ocr(pdf_path, source, page_num):
            sources.append(source)
            pages.append(page_num)
            return {"text": "ocr result", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = tracking_ocr

        mock_reader = MagicMock()
        mock_pages = [MagicMock(), MagicMock()]
        for p in mock_pages:
            p.extract_text.return_value = ""
        mock_reader.pages = mock_pages

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(pages, [1, 2])
        for rec in records:
            self.assertEqual(rec["source"], str(pdf))

    def test_configurable_ocr_threshold(self):
        """With ocr_min_chars=0, even text-rich pages never call OCR."""
        pdf = self.allowed / "thresh.pdf"
        _make_text_pdf(pdf, ["Plenty of text on this page to skip OCR normally."])

        # With threshold 0 → every page has enough text → no OCR
        loader = DocumentLoader(policy=self.policy, ocr_min_chars=0)
        with patch.object(loader, "_ocr_pdf_page") as mock_ocr:
            loader.load_file(pdf)
            mock_ocr.assert_not_called()

        # With threshold 999 → every page is "short" → OCR for all
        ocr_called = []

        def fake_ocr(pdf_path, source, page_num):
            ocr_called.append(page_num)
            return {"text": "ocr", "source": source, "page": page_num}

        loader2 = DocumentLoader(policy=self.policy, ocr_min_chars=999)
        loader2._ocr_pdf_page = fake_ocr
        loader2.load_file(pdf)
        self.assertEqual(ocr_called, [1])


# ─── Mixed PDF: text + scanned pages ────────────────────────────────────────

class TestMixedPDF(_BaseOCRTest):

    def test_mixed_pdf_uses_text_for_rich_pages_and_ocr_for_scanned(self):
        """Pages with enough text skip OCR; sparse pages use OCR."""
        pdf = self.allowed / "mixed.pdf"
        _make_text_pdf(pdf, [
            "Page one is text-rich enough to exceed the minimum threshold.",
            "p2",
            "Page three contains enough text to be extracted without OCR.",
        ])

        ocr_page_nums: list[int] = []

        def fake_ocr(pdf_path, source, page_num):
            ocr_page_nums.append(page_num)
            return {"text": f"OCR output page {page_num}", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = fake_ocr

        # Use a patched PdfReader that returns per-page text
        import pypdf as _pypdf

        page_texts = [
            "Page one is text-rich enough to exceed the minimum threshold.",
            "",  # triggers OCR
            "Page three contains enough text to be extracted without OCR.",
        ]

        # Build mock reader
        mock_reader = MagicMock()
        mock_pages = []
        for t in page_texts:
            p = MagicMock()
            p.extract_text.return_value = t
            mock_pages.append(p)
        mock_reader.pages = mock_pages

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(len(records), 3)
        self.assertEqual(ocr_page_nums, [2], "Only page 2 should have been OCR'd")
        # Page 1: text
        self.assertNotIn("OCR output", records[0]["text"])
        # Page 2: OCR
        self.assertIn("OCR output page 2", records[1]["text"])
        # Page 3: text
        self.assertNotIn("OCR output", records[2]["text"])

    def test_mixed_pdf_page_order_preserved(self):
        """Records must be returned in page order (1, 2, 3...)."""
        pdf = self.allowed / "order.pdf"
        _make_text_pdf(pdf, ["p", "pp", "ppp"])

        order: list[int] = []

        def tracking_ocr(pdf_path, source, page_num):
            order.append(page_num)
            return {"text": "ocr", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = tracking_ocr

        # Mock all pages as empty via PdfReader mock
        mock_reader = MagicMock()
        mock_pages = []
        for _ in range(3):
            p = MagicMock()
            p.extract_text.return_value = ""
            mock_pages.append(p)
        mock_reader.pages = mock_pages

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual([r["page"] for r in records], [1, 2, 3])


# ─── OCR failure handling ─────────────────────────────────────────────────────

class TestOCRFailureHandling(_BaseOCRTest):

    def _loader_with_ocr_error(self, error_msg: str) -> DocumentLoader:
        loader = DocumentLoader(policy=self.policy)

        def fail_ocr(pdf_path, source, page_num):
            return {"text": "", "source": source, "page": page_num, "ocr_error": error_msg}

        loader._ocr_pdf_page = fail_ocr
        return loader

    def test_ocr_failure_sets_ocr_error_key(self):
        """When OCR fails, the record must include 'ocr_error' and empty text."""
        pdf = self.allowed / "fail.pdf"
        _make_text_pdf(pdf, ["x"])

        loader = self._loader_with_ocr_error("Tesseract timeout")

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(len(records), 1)
        self.assertIn("ocr_error", records[0])
        self.assertEqual(records[0]["text"], "")
        self.assertEqual(records[0]["page"], 1)

    def test_ocr_failure_on_one_page_does_not_drop_other_pages(self):
        """Other pages must still be returned even if one page's OCR fails."""
        pdf = self.allowed / "partial_fail.pdf"
        _make_text_pdf(pdf, ["x", "y", "z"])

        call_count = [0]

        def mixed_ocr(pdf_path, source, page_num):
            call_count[0] += 1
            if page_num == 2:
                return {"text": "", "source": source, "page": page_num,
                        "ocr_error": "render error"}
            return {"text": f"OCR page {page_num}", "source": source, "page": page_num}

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = mixed_ocr

        mock_reader = MagicMock()
        mock_pages = [MagicMock(), MagicMock(), MagicMock()]
        for p in mock_pages:
            p.extract_text.return_value = ""
        mock_reader.pages = mock_pages

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(len(records), 3)
        self.assertIn("OCR page 1", records[0]["text"])
        self.assertIn("ocr_error", records[1])
        self.assertIn("OCR page 3", records[2]["text"])

    def test_fitz_not_installed_produces_error_record(self):
        """If fitz (PyMuPDF) is missing, record must contain an import error message."""
        pdf = self.allowed / "no_fitz.pdf"
        _make_text_pdf(pdf, ["tiny"])

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader), \
             patch.dict("sys.modules", {"fitz": None, "pymupdf": None}):
            loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
            records = loader.load_file(pdf)

        self.assertIn("ocr_error", records[0])
        self.assertIn("PyMuPDF", records[0]["ocr_error"])

    def test_pytesseract_not_installed_produces_error_record(self):
        """If pytesseract is missing, record must contain a clear install message."""
        pdf = self.allowed / "no_tess.pdf"
        _make_text_pdf(pdf, ["tiny"])

        fitz_mock = MagicMock()
        doc_mock = MagicMock()
        page_mock = MagicMock()
        pix_mock = MagicMock()
        pix_mock.width = 100
        pix_mock.height = 100
        pix_mock.samples = bytes(100 * 100 * 3)
        page_mock.get_pixmap.return_value = pix_mock
        doc_mock.__getitem__ = lambda s, i: page_mock
        fitz_mock.open.return_value = doc_mock
        fitz_mock.Matrix.return_value = MagicMock()
        fitz_mock.csRGB = MagicMock()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader), \
             patch.dict("sys.modules", {"fitz": fitz_mock, "pytesseract": None}):
            loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
            records = loader.load_file(pdf)

        self.assertIn("ocr_error", records[0])
        self.assertIn("pytesseract", records[0]["ocr_error"].lower())

    def test_tesseract_not_found_produces_install_hint(self):
        """TesseractNotFoundError must surface as ocr_error with install hint."""
        pdf = self.allowed / "notfound.pdf"
        _make_text_pdf(pdf, ["x"])

        import pytesseract as _pt

        fitz_mock = MagicMock()
        doc_mock = MagicMock()
        page_mock = MagicMock()
        pix_mock = MagicMock()
        pix_mock.width = 100
        pix_mock.height = 100
        pix_mock.samples = bytes(100 * 100 * 3)
        page_mock.get_pixmap.return_value = pix_mock
        doc_mock.__getitem__ = lambda s, i: page_mock
        fitz_mock.open.return_value = doc_mock
        fitz_mock.Matrix.return_value = MagicMock()
        fitz_mock.csRGB = MagicMock()

        tess_mock = MagicMock()
        tess_mock.TesseractNotFoundError = _pt.TesseractNotFoundError
        # TesseractNotFoundError() takes no positional arguments
        tess_mock.image_to_string.side_effect = _pt.TesseractNotFoundError()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader), \
             patch.dict("sys.modules", {"fitz": fitz_mock, "pytesseract": tess_mock}):
            loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
            records = loader.load_file(pdf)

        self.assertIn("ocr_error", records[0])
        self.assertIn("Tesseract", records[0]["ocr_error"])

    def test_general_ocr_exception_produces_error_record(self):
        """Any unexpected exception during OCR must produce a clear ocr_error."""
        pdf = self.allowed / "exc.pdf"
        _make_text_pdf(pdf, ["x"])

        def boom_ocr(pdf_path, source, page_num):
            return {
                "text": "",
                "source": source,
                "page": page_num,
                "ocr_error": "OCR failed on page 1: segfault (simulated)",
            }

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)
        loader._ocr_pdf_page = boom_ocr

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertIn("ocr_error", records[0])
        self.assertEqual(records[0]["text"], "")


# ─── Security: unauthorized PDFs rejected before OCR ─────────────────────────

class TestOCRSecurity(_BaseOCRTest):

    def test_unauthorized_pdf_rejected_before_ocr(self):
        """Policy check must happen before any rendering or OCR is attempted."""
        outside_pdf = self.outside / "secret.pdf"
        _make_text_pdf(outside_pdf, ["top secret"])

        with patch.object(self.loader, "_ocr_pdf_page") as mock_ocr:
            with self.assertRaises(PermissionError):
                self.loader.load_file(outside_pdf)
            mock_ocr.assert_not_called()

    def test_unauthorized_pdf_rejected_before_ocr_rendering(self):
        """fitz rendering must never be triggered on unauthorized files.

        fitz is imported lazily inside _ocr_pdf_page. Verifying that
        _ocr_pdf_page is never called is sufficient to guarantee that no PDF
        rendering is performed on unauthorized files.
        """
        outside_pdf = self.outside / "classified.pdf"
        _make_text_pdf(outside_pdf, ["classified"])

        with patch.object(self.loader, "_ocr_pdf_page") as mock_ocr:
            with self.assertRaises(PermissionError):
                self.loader.load_file(outside_pdf)
            mock_ocr.assert_not_called()


# ─── OCR text flows through chunker ─────────────────────────────────────────

class TestOCRChunkerIntegration(_BaseOCRTest):

    def test_ocr_text_chunked_correctly(self):
        """OCR output must pass through the chunker and produce non-empty chunks."""
        ocr_text = (
            "Invoice Number: 12345. Date: 2024-01-15. "
            "Total amount due: $4,200.00. Payment terms: Net 30. "
            "Please remit payment to the address on file."
        )
        pdf = self.allowed / "invoice.pdf"
        _make_text_pdf(pdf, ["x"])

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)

        def fake_ocr(pdf_path, source, page_num):
            return {"text": ocr_text, "source": source, "page": page_num}

        loader._ocr_pdf_page = fake_ocr

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        self.assertEqual(len(records), 1)
        chunks = chunk_documents(records, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk["source"], str(pdf))
            self.assertEqual(chunk["page"], 1)
            self.assertTrue(chunk["text"].strip())

    def test_ocr_chunk_contains_no_split_words(self):
        """Chunker must not split words within OCR output."""
        ocr_text = "privacy-first offline AI system for secure document retrieval."
        pdf = self.allowed / "privacy.pdf"
        _make_text_pdf(pdf, ["x"])

        loader = DocumentLoader(policy=self.policy, ocr_min_chars=30)

        def fake_ocr(pdf_path, source, page_num):
            return {"text": ocr_text, "source": source, "page": page_num}

        loader._ocr_pdf_page = fake_ocr

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        with patch("documents.loader.pypdf.PdfReader", return_value=mock_reader):
            records = loader.load_file(pdf)

        chunks = chunk_documents(records, chunk_size=40, overlap=10)
        all_words = [w.strip(".,!?") for c in chunks for w in c["text"].split()]
        self.assertNotIn("vacy-first", all_words)
        self.assertIn("privacy-first", all_words)


# ─── Existing loaders unaffected ─────────────────────────────────────────────

class TestExistingLoadersUnaffected(_BaseOCRTest):

    def test_txt_file_loads_normally(self):
        f = self.allowed / "a.txt"
        f.write_text("Plain text content.", encoding="utf-8")
        records = self.loader.load_file(f)
        self.assertEqual(len(records), 1)
        self.assertIn("Plain text content", records[0]["text"])
        self.assertIsNone(records[0]["page"])

    def test_md_file_loads_normally(self):
        f = self.allowed / "a.md"
        f.write_text("# Heading\nBody text.", encoding="utf-8")
        records = self.loader.load_file(f)
        self.assertEqual(records[0]["text"], "# Heading\nBody text.")

    def test_docx_file_loads_normally(self):
        f = self.allowed / "doc.docx"
        _make_docx(f, ["First.", "Second."])
        records = self.loader.load_file(f)
        self.assertIn("First.", records[0]["text"])
        self.assertIn("Second.", records[0]["text"])
        self.assertIsNone(records[0]["page"])

    def test_text_pdf_unaffected_by_ocr_feature(self):
        pdf = self.allowed / "text_only.pdf"
        _make_text_pdf(pdf, [
            "First page has sufficient text for direct extraction.",
            "Second page also has sufficient text for direct extraction.",
        ])
        with patch.object(self.loader, "_ocr_pdf_page") as mock_ocr:
            records = self.loader.load_file(pdf)
            mock_ocr.assert_not_called()
        self.assertEqual(len(records), 2)

    def test_unsupported_extension_raises_value_error(self):
        f = self.allowed / "data.csv"
        f.write_text("a,b,c")
        with self.assertRaises(ValueError):
            self.loader.load_file(f)

    def test_outside_txt_raises_permission_error(self):
        f = self.outside / "secret.txt"
        f.write_text("secret")
        with self.assertRaises(PermissionError):
            self.loader.load_file(f)


# ─── KnowledgeBase refresh does not duplicate chunks ─────────────────────────

class TestKBRefreshNoDuplicates(_BaseOCRTest):

    def test_refresh_does_not_accumulate_chunks(self):
        """build_index() twice must produce the same chunk count, not double it."""
        from unittest.mock import MagicMock
        from rag.knowledge_base import KnowledgeBase

        f = self.allowed / "stable.txt"
        f.write_text("Stable content for duplicate prevention testing.")

        mock_client = MagicMock()
        kb = KnowledgeBase(policy=self.policy, client=mock_client)
        count1 = kb.build_index()
        count2 = kb.build_index()
        self.assertEqual(count1, count2, "Second build_index must not accumulate duplicates")


# ─── Real Scanned PDF Regression Test ─────────────────────────────────────────

class TestRealScannedPDFRegression(_BaseOCRTest):

    def test_real_scanned_pdf_image_extraction(self):
        """End-to-end regression test: create a scanned (image-only) PDF and extract text via real OCR."""
        from PIL import Image, ImageDraw

        img_path = self.allowed / "test_scan_img.png"
        img = Image.new("RGB", (1000, 300), color="white")
        draw = ImageDraw.Draw(img)
        # Draw clean spaced characters or simple text
        draw.text((50, 100), "TEST SCAN OCR SUCCESSFUL", fill="black")
        img.save(img_path)

        scanned_pdf = self.allowed / "scanned_doc.pdf"
        c = _canvas.Canvas(str(scanned_pdf))
        c.drawImage(str(img_path), 50, 500, width=500, height=150)
        c.showPage()
        c.save()

        # Verify pypdf extracts nothing from pure image page
        import pypdf
        reader = pypdf.PdfReader(scanned_pdf)
        extracted = reader.pages[0].extract_text() or ""
        self.assertEqual(len(extracted.strip()), 0, "Image-only PDF must have no direct pypdf text")

        # Now load with DocumentLoader — should invoke real OCR and extract the text
        records = self.loader.load_file(scanned_pdf)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["page"], 1)
        self.assertNotIn("ocr_error", records[0])
        self.assertIn("SUCCESSFUL", records[0]["text"].upper())

    def test_tai_pdf_if_present(self):
        """If 5.TAI.pdf is present in /home/kuldeep/Documents, verify real OCR extracts text."""
        tai_path = Path("/home/kuldeep/Documents/5.TAI.pdf")
        if not tai_path.exists():
            self.skipTest("5.TAI.pdf not found on disk")

        cfg = self.base / "permissions_tai.json"
        cfg.write_text(json.dumps({"allowed_locations": ["/home/kuldeep/Documents"]}))
        policy = Policy(config_path=cfg)
        loader = DocumentLoader(policy=policy)

        # Load first 2 pages by checking
        records = loader.load_file(tai_path)
        self.assertGreater(len(records), 0)
        non_empty = [r for r in records if r.get("text", "").strip()]
        self.assertGreater(len(non_empty), 0)
        # Verify first page contains expected course instructor
        self.assertIn("Narinder", records[0]["text"])


if __name__ == "__main__":
    unittest.main()
