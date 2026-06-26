"""
retriever.py
Week 1: basic top-k retrieval from ChromaDB
Week 2: adds Cohere reranking to compress 15 chunks → top 3
"""


from src.embedder import embed_query
from src.vectorstore import query_chunks

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


def retrieve_and_rerank(query: str, top_k_retrieve: int = 15, top_k_final: int = 3) -> list[dict]:
    """
    Step 1: retrieve top 15 chunks from ChromaDB/Qdrant (broad net)
    Step 2: Cohere Rerank compresses them to top 3 (precision)
    """
    import os
    import cohere
    from dotenv import load_dotenv
    load_dotenv()

    # broad retrieval of vector database chunks, top 15 by default
    query_embedding = embed_query(query)
    chunks = query_chunks(query_embedding, top_k=top_k_retrieve)
    texts = [c["text"] for c in chunks]

    # reranking process of the chunks retrieved from the first step
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_k_final
    )

    reranked = []
    for r in rerank_results.results:
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


if __name__ == "__main__":
    # Week 1 test
    results = retrieve("What was the revenue growth?")
    print("\nTop chunk text:")
    print(results[0]["text"])