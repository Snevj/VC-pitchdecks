"""
loader.py
Extracts raw text from a PDF file, page by page.
"""

import os
import unicodedata
import pymupdf
from langsmith_helper import get_traceable
traceable = get_traceable()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# this is basically the document parsing and extraction module that reads a PDF file, extracts text
def load_pdf(pdf_path: str = None, filename: str = None) -> list[dict]:
    """
    Load a PDF and extract text page by page.

    Pass either:
      - pdf_path: a full/absolute path to any PDF, OR
      - filename: just the filename, resolved against UPLOADS_DIR
    """
    if pdf_path is None:
        if filename is None:
            raise ValueError("Provide either pdf_path or filename")
        pdf_path = os.path.join(UPLOADS_DIR, filename)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_num": page_num + 1,
                "text": text
            })

    print(f"Loaded {len(pages)} pages from {pdf_path}")
    return pages


# this function normalizes the extracted text by cleaning up unicode characters, fixing hyphenated...
def normalize(pages: list[dict]) -> list[dict]:
    for page in pages:
        text = page["text"]

        # 1. Normalize unicode characters (removes weird spacing and broken symbols)
        text = unicodedata.normalize("NFKC", text)

        # 2. Fix hyphenated line breaks (e.g., "para-\ngraph" -> "paragraph")
        # ... rest of your existing normalize logic continues here

# if this script is run directly, this function will execute
if __name__ == "__main__":
    raw_pages = load_pdf(filename="nestle_files.pdf")   # ← fixed: use filename, not undefined pdf_path
    normalized_pages = normalize(raw_pages)
    print("\n--- Cleaned Sample Output ---")
    print(normalized_pages[0]["text"][:500])  # Prints the first 500 characters