"""
retriever.py
Week 1: basic top-k retrieval from ChromaDB
Week 2: adds Cohere reranking to compress 15 chunks → top 3
Week 3: adds BM25 hybrid search, fused with vector search via RRF
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import embed_query
from src.vectorstore import query_chunks
from src.bm25_index import query_chunks_bm25


# Project-wide tracing decorator
from langsmith_helper import get_traceable
traceable = get_traceable()

@traceable(name="retrieve")
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Embeds the query, searches ChromaDB, returns top_k chunks.
    """
    query_embedding = embed_query(query)
    chunks = query_chunks(query_embedding, top_k=top_k)

    print(f"\nRetrieved {len(chunks)} chunks for: '{query}'")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] page {c['page_num']} | score {c['score']} | {c['text'][:80]}...")

    return chunks


def reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:  # NEW function
    """
    Merges vector-search and BM25 results by rank position, not raw score
    (their scores live on different scales, so RRF is the standard way to combine them).
    """
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(vector_results):
        key = (chunk["page_num"], chunk["text"][:50])  # dedup key for chunks appearing in both lists
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = (chunk["page_num"], chunk["text"][:50])
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_lookup[key] for key, _ in fused]


@traceable(name="retrieve_and_rerank")
def retrieve_and_rerank(
    query: str,
    top_k_retrieve: int = 15,
    top_k_final: int = 4,
    relevance_threshold: float = 0.3,
) -> list[dict]:
    """
    Step 1: retrieve top 15 chunks from vector search (Qdrant) AND BM25 keyword search,
             fuse the two ranked lists via Reciprocal Rank Fusion (broad net, hybrid recall)
    Step 2: Cohere Rerank scores all candidates; chunks below
            relevance_threshold are dropped before slicing to top_k_final.
            Falls back to the single best chunk if everything is below threshold,
            so the LLM never receives an empty context.
    """
    import os
    import cohere
    from dotenv import load_dotenv
    load_dotenv()

    # broad retrieval — hybrid of vector search and keyword (BM25) search
    query_embedding = embed_query(query)
    vector_chunks = query_chunks(query_embedding, top_k=top_k_retrieve)
    bm25_chunks = query_chunks_bm25(query, top_k=top_k_retrieve)
    chunks = reciprocal_rank_fusion(vector_chunks, bm25_chunks)[:top_k_retrieve]
    texts = [c["text"] for c in chunks]

    # reranking process — request scores for ALL candidates, not just top_k_final,
    # so we can filter by relevance before truncating
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=len(texts)
    )

    # filter out low-relevance chunks first, then take top_k_final of what's left
    filtered = [r for r in rerank_results.results if r.relevance_score >= relevance_threshold]

    if not filtered:
        # nothing cleared the bar — fall back to the single best chunk
        # rather than handing the LLM an empty context
        filtered = [rerank_results.results[0]]
        print(f"  ⚠️  No chunks cleared threshold {relevance_threshold}; falling back to best match (score {filtered[0].relevance_score:.4f})")

    selected = filtered[:top_k_final]

    reranked = []
    for r in selected:
        original_chunk = chunks[r.index]
        reranked.append({
            "text": original_chunk["text"],
            "page_num": original_chunk["page_num"],
            "score": round(r.relevance_score, 4),
            "original_rank": r.index
        })

    print(f"\nReranked to {len(reranked)} chunks for: '{query}'")
    for i, c in enumerate(reranked):
        print(f"  [{i+1}] page {c['page_num']} | rerank score {c['score']} | {c['text'][:80]}...")

    return reranked

# if __name__ == "__main__":
#     # Week 1 test
#     results = retrieve("What was the revenue growth?")
#     print("\nTop chunk text:")
#     print(results[0]["text"])
if __name__ == "__main__":
    # Week 1 test
    results = retrieve("What was the revenue growth?")
    print("\nTop chunk text:")
    print(results[0]["text"])

    # NEW — test the hybrid + rerank pipeline
    print("\n--- Testing retrieve_and_rerank (hybrid + Cohere) ---")
    hybrid_results = retrieve_and_rerank("What was the incremental financial impact of the New Labour Codes?")