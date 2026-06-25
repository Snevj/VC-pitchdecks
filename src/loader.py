"""
loader.py
Extracts raw text from a PDF file, page by page.
"""

import os
import unicodedata
import pymupdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "myfile.pdf")


def load_pdf(pdf_path: str) -> list[dict]:
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


def normalize(pages: list[dict]) -> list[dict]:
    for page in pages:
        text = page["text"]
        
        # 1. Normalize unicode characters (removes weird spacing and broken symbols)
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Fix hyphenated line breaks (e.g., "para-\ngraph" -> "paragraph")
        text = text.replace("-\n", "").replace("- \n", "")
        
        # 3. Standardize curly quotes to standard straight quotes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        
        page["text"] = text
    return pages


if __name__ == "__main__":
    raw_pages = load_pdf(pdf_path)
    normalized_pages = normalize(raw_pages)
    print("\n--- Cleaned Sample Output ---")
    print(normalized_pages[0]["text"][:500])  # Prints the first 500 chars of cleaned text