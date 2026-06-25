"""
vectorstore.py
Stores chunk embeddings in ChromaDB and retrieves similar chunks for a query.
ChromaDB runs locally — no cloud, no cost.
"""
import os
import sys

# Add project root to imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared config values.
from config import VECTOR_DB_TYPE, QDRANT_URL, EMBED_MODEL

COLLECTION_NAME = "financial_docs"  # str
VECTOR_SIZE = 384  # int for MiniLM vectors

_client = None

def get_db_client():
    """Return the cached vector DB client."""
    global _client
    if _client is not None:
        return _client

    if VECTOR_DB_TYPE == "chroma":
        import chromadb
        print("Initializing Local ChromaDB Engine...")
        _client = chromadb.PersistentClient(path="./chroma_db")
        
    elif VECTOR_DB_TYPE == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        # Path = local. URL = remote.
        if QDRANT_URL.startswith("http"):
            print(f"Connecting to Qdrant Server over network at: {QDRANT_URL}")
            _client = QdrantClient(url=QDRANT_URL)
        else:
            print("Running Qdrant in Local Embedded Mode...")
            _client = QdrantClient(path="./qdrant_db")
            
        # Create the collection once.
        if not _client.collection_exists(collection_name=COLLECTION_NAME):
            from qdrant_client.models import Distance, VectorParams
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            
    return _client

def add_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """
    Store chunk text and vectors.
    chunks: list[dict]
    embeddings: list[list[float]]
    """
    from qdrant_client.models import PointStruct

    client = get_db_client()
    points = []

    # Turn each chunk into a Qdrant point.
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Fallback id if chunk_id is missing.
        point_id = chunk.get("chunk_id", idx)
        
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "page_num": chunk["page_num"]
                }
            )
        )

    # Bulk write.
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(chunks)} chunks in Qdrant.")

#top-7 chunks which are most suitable for the query are returned by this function
def query_chunks(query_embedding: list[float], top_k: int = 7) -> list[dict]:
    """
    Return the top matches for one query vector.
    """
    client = get_db_client()

    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points

    matched_chunks = []
    for point in search_results:
        matched_chunks.append({
            "text": point.payload["text"],
            "page_num": point.payload["page_num"],
            "score": round(point.score, 4)  # float
        })

    return matched_chunks


def clear_collection():
    """Delete the active collection."""
    client = get_db_client()
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' cleared.")


if __name__ == "__main__":
    # Small local test.
    from embedder import embed_texts, embed_query

    # Sample chunks.
    test_chunks = [
        {"chunk_id": 0, "page_num": 1, "text": "Revenue grew 12% year over year."},
        {"chunk_id": 1, "page_num": 1, "text": "Operating expenses increased by 5%."},
        {"chunk_id": 2, "page_num": 2, "text": "The company faces regulatory risk in Europe."},
    ]
    
    print("--- Testing Integrated Qdrant Vector Engine Pipeline ---")
    embeddings = embed_texts([c["text"] for c in test_chunks])
    add_chunks(test_chunks, embeddings)

    q_emb = embed_query("What was the revenue growth?")
    results = query_chunks(q_emb, top_k=2)
    
    print("\n Matching Results Pulled:")
    for r in results:
        print(r)
    if _client:
        _client.close()