"""
bm25_index.py
Keyword-based (BM25) retrieval index, run in parallel with Qdrant
for hybrid search. Persisted separately from the vector store.
"""
import os
import pickle
from rank_bm25 import BM25Okapi

from src.langsmith_helper import get_traceable
traceable = get_traceable()

BM25_INDEX_PATH = "./bm25_index.pkl"
_bm25_cache = None


@traceable(name="BM25 Build Index")
def build_bm25_index(chunks: list[dict]):
    """Builds/updates the BM25 index, merging with any previously indexed chunks."""
    existing_chunks = []
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            existing_chunks = pickle.load(f)["chunks"]

    all_chunks = existing_chunks + chunks
    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

    global _bm25_cache
    _bm25_cache = (bm25, all_chunks)
    print(f"✅ BM25 index updated with {len(all_chunks)} total chunks.")


def get_bm25_index():
    global _bm25_cache
    if _bm25_cache is None:
        if not os.path.exists(BM25_INDEX_PATH):
            return None, []
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25_cache = (data["bm25"], data["chunks"])
    return _bm25_cache


@traceable(name="BM25 Query Chunks")
def query_chunks_bm25(query: str, top_k: int = 15) -> list[dict]:
    bm25, chunks = get_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {**chunks[i], "score": round(float(scores[i]), 4)}
        for i in ranked_idx
    ]


def clear_bm25_index():
    global _bm25_cache
    if os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)
    _bm25_cache = None
    print("BM25 index cleared successfully.")