"""
chunker.py
Splits clean raw text into semantic overlapping chunks for vector storage.
"""

import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MODEL_CHUNKER

# Keep project root relative path mappings clean
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#this text splitter uses a recursive character-based approach to split text into chunks based on token boundaries, ensuring that the chunks are semantically meaningful and maintain context. The chunk size and overlap can be adjusted to suit specific use cases, such as preparing text for embedding or indexing in a vector database.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Takes a list of page dictionaries, splits the text recursively,
    and returns a list of chunk dictionaries with correct page metadata.
    """
    # 🛠️ Re-verify you are using a standard encoding name tiktoken recognizes
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "•", " ", ""]
    )

    all_chunks = []

    # 🛠️ Fix: Iterate through the pages list and split each page individually
    for page in pages:
        raw_text = page["text"]
        page_num = page["page_num"]

        # Split this single page's string text into smaller string chunks
        string_chunks = text_splitter.split_text(raw_text)

        # Repackage them back into your pipeline's dictionary structure
        for chunk_string in string_chunks:
            all_chunks.append({
                "text": chunk_string,
                "page_num": page_num
            })

    return all_chunks

#further cleaning of the text can be done here if needed, such as removing extra whitespace, special characters, etc.
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
    text_chunks = chunk_text(clean_docs)
    
    # 3. Print Results
    print(f"Total Chunks Generated: {len(text_chunks)}\n")
    for idx, chunk in enumerate(text_chunks):
        print(f" [CHUNK {idx + 1}] (Length: {len(chunk)} chars):")
        print(chunk)
        print("=" * 50)