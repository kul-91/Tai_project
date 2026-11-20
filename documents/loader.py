"""Secure document loader for SentinelAI.

Supports TXT, Markdown, PDF (with OCR fallback for scanned pages), and DOCX.
All access is gated by the security.Policy authorization layer.

OCR pipeline (PDF only)
-----------------------
1. Each PDF page is extracted normally via pypdf.
2. If the extracted text is shorter than `ocr_min_chars` (default 30), the page
   is considered image-only or scan-based.
3. Scanned pages are rendered to a PIL image via PyMuPDF (fitz) at `ocr_dpi` DPI
   (default 200) and passed to pytesseract for OCR.
4. Only pages with too little direct text trigger OCR — text-rich pages are never
   rendered, preventing duplicate text from embedded OCR layers.
5. If OCR fails on a page the record text is set to the empty string and a
   "ocr_error" key is added to the record with the error message.
   The page record is still returned so callers can distinguish "page existed but
   was unreadable" from "page was never attempted".
6. OCR requires system packages (Tesseract + language data). See INSTALL section.

INSTALL (CachyOS / Arch Linux)
------------------------------
    sudo pacman -S tesseract tesseract-data-eng

Verify:
    tesseract --version
    tesseract --list-langs   # should include 'eng'

Python packages (virtual-env):
    pip install PyMuPDF pytesseract

Importing without Tesseract installed
--------------------------------------
pytesseract and fitz are imported lazily inside the OCR path.  The module loads
fine even if they are absent; OCR will fail at runtime with a clear error message
that surfaces in the record's "ocr_error" field rather than crashing the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import docx
import pypdf

from security.policy import Policy

SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ""}

# ─── OCR defaults ────────────────────────────────────────────────────────────

# Pages whose extracted text (stripped) is shorter than this threshold are
# considered image-only / scanned and will be processed with OCR.
OCR_MIN_CHARS: int = 30

# Resolution used when rendering a PDF page to an image for OCR.
# 200 DPI gives a good balance between OCR accuracy and memory usage.
# Increase to 300 for dense text; decrease to 150 for faster processing.
OCR_DEFAULT_DPI: int = 200

# Tesseract language string.  Multiple languages: "eng+fra"
OCR_DEFAULT_LANG: str = "eng"


# ─── DocumentLoader ───────────────────────────────────────────────────────────

class DocumentLoader:
    """Secure document loader for SentinelAI.

    Reads local document files strictly within permitted filesystem locations
    enforced by security.Policy.

    Args:
        policy:        Security Policy instance. Defaults to Policy().
        ocr_dpi:       Resolution (DPI) used when rendering PDF pages for OCR.
                       Default: 200.
        ocr_lang:      Tesseract language string, e.g. "eng" or "eng+fra".
                       Default: "eng".
        ocr_min_chars:  Minimum number of meaningful characters a page must
                       contain after pypdf extraction to skip OCR.
                       Default: 30.
        tessdata_dir:  Optional path to the tessdata directory containing
                       traineddata files. If None, auto-discovers from
                       TESSDATA_PREFIX, project directory, or ~/.local/share/tessdata.
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        ocr_dpi: int = OCR_DEFAULT_DPI,
        ocr_lang: str = OCR_DEFAULT_LANG,
        ocr_min_chars: int = OCR_MIN_CHARS,
        tessdata_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.policy = policy if policy is not None else Policy()
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang
        self.ocr_min_chars = ocr_min_chars
        self.tessdata_dir = tessdata_dir

    def _resolve_tessdata_dir(self) -> Optional[Path]:
        """Resolve a valid tessdata directory containing the target language data."""
        candidates = []
        if self.tessdata_dir:
            candidates.append(Path(self.tessdata_dir).expanduser().resolve())

        import os
        if "TESSDATA_PREFIX" in os.environ:
            candidates.append(Path(os.environ["TESSDATA_PREFIX"]).expanduser().resolve())

        # Project root tessdata
        candidates.append(Path(__file__).resolve().parent.parent / "tessdata")
        # User local tessdata
        candidates.append(Path.home() / ".local" / "share" / "tessdata")

        for path in candidates:
            if path.is_dir():
                lang_file = path / f"{self.ocr_lang}.traineddata"
                if lang_file.exists() or any(path.glob("*.traineddata")):
                    return path
        return None

    def _get_tesseract_config(self) -> str:
        """Build configuration string for pytesseract with tessdata-dir if available."""
        resolved = self._resolve_tessdata_dir()
        if resolved:
            return f'--tessdata-dir "{resolved}"'
        return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def load_file(self, path: Union[str, Path]) -> list[dict[str, Any]]:
        """Load and extract text from a permitted document file.

        Args:
            path: Path to the document file.

        Returns:
            List of document record dicts::

                [
                    {
                        "text":     str,            # extracted / OCR'd text
                        "source":   str,            # absolute path string
                        "page":     Optional[int],  # 1-based page number (PDF)
                        # only present on OCR failure:
                        "ocr_error": str,
                    }
                ]

        Raises:
            PermissionError:   If path access is denied by policy.
            ValueError:        If file extension is unsupported.
            FileNotFoundError: If path does not exist.
            IsADirectoryError: If path is a directory.
        """
        # SECURITY MANDATE: Perform policy check BEFORE reading anything
        self.policy.check_access(path)

        target_path = Path(path).expanduser().resolve()

        if not target_path.exists():
            raise FileNotFoundError(f"File not found: '{path}'")
        if target_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: '{path}'")

        ext = target_path.suffix.lower()
        if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise ValueError(
                f"Unsupported document extension '{ext}'. "
                f"Supported formats: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
            )

        source_str = str(target_path)

        if ext in {".txt", ".md", ""}:
            if ext == "":
                try:
                    with open(target_path, "rb") as bf:
                        if b"\x00" in bf.read(1024):
                            raise ValueError(f"Unsupported binary document without extension: '{path}'")
                except OSError:
                    pass
            return self._load_text_file(target_path, source_str)
        elif ext == ".pdf":
            return self._load_pdf_file(target_path, source_str)
        elif ext == ".docx":
            return self._load_docx_file(target_path, source_str)

        return []  # unreachable; satisfies type checker

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_text_file(self, path: Path, source: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [{"text": text, "source": source, "page": None}]

    def _load_pdf_file(self, path: Path, source: str) -> list[dict[str, Any]]:
        """Load a PDF, applying page-level OCR fallback for scanned pages.

        Text-based pages are extracted normally via pypdf.  Pages whose
        extracted text is shorter than `ocr_min_chars` characters (after
        stripping whitespace) are rendered via PyMuPDF and passed to
        Tesseract for OCR.
        """
        records: list[dict[str, Any]] = []

        try:
            reader = pypdf.PdfReader(path)
            num_pages = len(reader.pages)
        except Exception as exc:
            # Return a single error record rather than crashing the pipeline
            return [{"text": "", "source": source, "page": None,
                     "ocr_error": f"Failed to open PDF: {exc}"}]

        for idx in range(num_pages):
            page_num = idx + 1
            try:
                page_text: str = reader.pages[idx].extract_text() or ""
            except Exception:
                page_text = ""

            if len(page_text.strip()) >= self.ocr_min_chars:
                # Sufficient direct text — no OCR needed
                records.append({"text": page_text, "source": source, "page": page_num})
            else:
                # Attempt OCR on this page
                ocr_record = self._ocr_pdf_page(path, source, page_num)
                records.append(ocr_record)

        return records

    def _ocr_pdf_page(self, pdf_path: Path, source: str, page_num: int) -> dict[str, Any]:
        """Render a single PDF page and apply Tesseract OCR.

        Args:
            pdf_path: Absolute resolved path to the PDF.
            source:   Source string for the record.
            page_num: 1-based page number.

        Returns:
            A document record dict.  "ocr_error" is set on failure.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            try:
                import pymupdf as fitz
            except ImportError:
                return {
                    "text": "",
                    "source": source,
                    "page": page_num,
                    "ocr_error": (
                        f"OCR failed on page {page_num}: PyMuPDF (fitz/pymupdf) is not installed. "
                        "Run: pip install PyMuPDF"
                    ),
                }

        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return {
                "text": "",
                "source": source,
                "page": page_num,
                "ocr_error": (
                    f"OCR failed on page {page_num}: pytesseract or Pillow is not installed. "
                    "Run: pip install pytesseract pillow"
                ),
            }

        # PyMuPDF page indices are 0-based
        fitz_doc = None
        try:
            fitz_doc = fitz.open(str(pdf_path))
            fitz_page = fitz_doc[page_num - 1]

            # Render page to a pixmap at the configured DPI
            zoom = self.ocr_dpi / 72.0  # 72 pt/inch is the PDF baseline
            mat = fitz.Matrix(zoom, zoom)
            pix = fitz_page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Run Tesseract with auto-resolved tessdata directory configuration
            tess_config = self._get_tesseract_config()
            ocr_text: str = pytesseract.image_to_string(img, lang=self.ocr_lang, config=tess_config)

            return {"text": ocr_text, "source": source, "page": page_num}

        except pytesseract.TesseractNotFoundError:
            return {
                "text": "",
                "source": source,
                "page": page_num,
                "ocr_error": (
                    f"OCR failed on page {page_num}: Tesseract executable not found. "
                    "Install it with: sudo pacman -S tesseract tesseract-data-eng"
                ),
            }
        except pytesseract.TesseractError as exc:
            return {
                "text": "",
                "source": source,
                "page": page_num,
                "ocr_error": f"OCR failed on page {page_num}: {exc}",
            }
        except Exception as exc:
            return {
                "text": "",
                "source": source,
                "page": page_num,
                "ocr_error": f"OCR failed on page {page_num}: {exc}",
            }
        finally:
            if fitz_doc is not None:
                fitz_doc.close()

    def _load_docx_file(self, path: Path, source: str) -> list[dict[str, Any]]:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        full_text = "\n".join(paragraphs)
        return [{"text": full_text, "source": source, "page": None}]
