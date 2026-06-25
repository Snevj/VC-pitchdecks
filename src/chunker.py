"""
chunker.py
Splits clean raw text into semantic overlapping chunks for vector storage.
"""

import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Keep project root relative path mappings clean
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def chunk_text(raw_text: str, chunk_size: int = 300, chunk_overlap: int = 40) -> list:
    """
    Slices raw text blocks into chunk sets using token boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", " ", ""]
    )
    return text_splitter.split_text(raw_text)

if __name__ == "__main__":
    print("--- Running Token-Based Chunker ---")
    
    # Import your successful loader functions
    from src.loader import load_pdf, normalize, pdf_path
    
    # 1. Pipeline Execution: Load -> Normalize
    raw_docs = load_pdf(pdf_path)
    clean_docs = normalize(raw_docs)
    
    # Extract text from the first page object
    raw_content = clean_docs[0]["text"]
    
    # 2. Slice Content
    text_chunks = chunk_text(raw_content)
    
    # 3. Print Results
    print(f"Total Chunks Generated: {len(text_chunks)}\n")
    for idx, chunk in enumerate(text_chunks):
        print(f"📦 [CHUNK {idx + 1}] (Length: {len(chunk)} chars):")
        print(chunk)
        print("=" * 50)