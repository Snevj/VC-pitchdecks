"""
chunker.py
Splits clean raw text into semantic overlapping chunks for vector storage.

Adds general header-detection: before running the standard recursive splitter,
this inserts a forced break marker before lines that look like section headers
(e.g. "Commodity Outlook:", "Financial Highlights:", "Key Risks"). This prevents
dense multi-topic chunks where an important section gets buried alongside
unrelated content — a real failure mode confirmed via testing, where it both
degraded LLM generation accuracy and degraded RAGAS judge faithfulness scoring.

This is a GENERAL heuristic (not hardcoded to any one document's specific
headers) so it should generalize across different financial documents.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← single import
from config import UPLOAD_DIR, COLLECTION_NAME

# A "header-like" line is short, doesn't end in normal sentence punctuation,
# and either ends in a colon or is in title/sentence case starting a paragraph.
# This regex looks for short lines (<80 chars) ending in a colon, which
# reliably matches patterns like "Commodity Outlook:", "Cautionary Statement:",
# "Key Highlights for the fourth quarter ended 31 March 2026:", etc.
HEADER_PATTERN = re.compile(r'^(?=.{1,80}$)([A-Z][A-Za-z0-9 ,/&\'\-]+:)\s*$', re.MULTILINE)


def _insert_header_breaks(text: str) -> str:
    """
    Scans for header-like lines and inserts extra blank lines before them,
    so the recursive splitter's '\\n\\n\\n' separator (highest priority)
    reliably breaks right before a new section starts.
    """
    def add_break(match):
        return "\n\n\n" + match.group(1)

    return HEADER_PATTERN.sub(add_break, text)


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Takes a list of page dictionaries, splits the text recursively,
    and returns a list of chunk dictionaries with correct page metadata.
    Section headers are detected generically and used as forced split
    boundaries to avoid burying important content inside dense, multi-topic
    chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,      # down from 512 — forces breaks at detected headers
        chunk_overlap=60,    # scaled down proportionally
        separators=["\n\n\n", "\n\n", "\n", "•", " ", ""]  # triple-newline = highest priority
    )

    all_chunks = []

    for page in pages:
        raw_text = page["text"]
        page_num = page["page_num"]

        # Insert forced breaks before detected section headers, then split normally
        prepared_text = _insert_header_breaks(raw_text)
        string_chunks = text_splitter.split_text(prepared_text)

        for chunk_string in string_chunks:
            if chunk_string.strip():                         # ← skip empty chunks
                all_chunks.append({
                    "text": chunk_string.strip(),
                    "page_num": page_num
                })

    print(f"✅ Chunked {len(pages)} pages → {len(all_chunks)} chunks "
          f"(size=300, overlap=60)")
    return all_chunks


if __name__ == "__main__":
    print("--- Running Token-Based Chunker (with header-aware splitting) ---")

    from src.loader import load_pdf, normalize  # ← removed pdf_path, use filename instead

    raw_docs = load_pdf(filename="nestle_files.pdf")   # ← fixed: uses UPLOADS_DIR
    clean_docs = normalize(raw_docs)

    text_chunks = chunk_text(clean_docs)

    print(f"\nTotal Chunks Generated: {len(text_chunks)}\n")
    for idx, chunk in enumerate(text_chunks):
        print(f"[CHUNK {idx + 1}] page {chunk['page_num']} ({len(chunk['text'])} chars):")
        print(chunk['text'][:200])
        print("=" * 50)