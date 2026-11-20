from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": page_number, "text": text})

    return pages