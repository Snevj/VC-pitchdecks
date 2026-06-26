"""
vectorstore.py
Stores chunk embeddings in ChromaDB and retrieves similar chunks for a query.
ChromaDB runs locally — no cloud, no cost.
"""
import os
import sys
from qdrant_client.http import models as rest_models
from qdrant_client import QdrantClient
# Add project root to imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DB_TYPE, QDRANT_URL, EMBED_MODEL, VECTOR_SIZE, COLLECTION_NAME

VECTOR_SIZE = 384  # int for MiniLM vectors

_client = None

def get_db_client():
    global _client
    
    # 🛠️ ONLY create a new client if one doesn't exist yet!
    if _client is None:
        if VECTOR_DB_TYPE == "qdrant":
            print("Running Qdrant in Local Embedded Mode...")
            _client = QdrantClient(path="./qdrant_db")
        else:
            # Your ChromaDB or other initialization logic here
            pass
            
    return _client
def add_chunks(chunks: list[dict], embeddings: list[list[float]]):
    client = get_db_client() # Or whatever your database initialization wrapper is called
    
    # ── 🛠️ ADD THIS SELF-HEALING COLLECTION CHECK ──────────────────────────
    # Check if the target collection exists in memory/storage
    collections_response = client.get_collections()
    existing_collections = [col.name for col in collections_response.collections]
    
    if COLLECTION_NAME not in existing_collections:
        print(f" Creating fresh Qdrant Collection: '{COLLECTION_NAME}' (384 Dimensions)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest_models.VectorParams(
                size=384,  # Matches all-MiniLM-L6-v2 vector dimension footprint
                distance=rest_models.Distance.COSINE
            )
        )
    # ─────────────────────────────────────────────────────────────────────────

    # Your existing code below handles building points and uploading...
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Generate a unique point id or use a UUID
        points.append(
            rest_models.PointStruct(
                id=i, 
                vector=embedding, 
                payload={"text": chunk["text"], "page_num": chunk["page_num"]}
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Successfully synchronized and stored {len(chunks)} chunks in database.")

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