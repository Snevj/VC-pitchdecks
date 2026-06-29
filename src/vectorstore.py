"""
vectorstore.py
Stores chunk embeddings in Qdrant Local Embedded Mode and retrieves similar chunks.
"""
import os
import sys
import uuid  # Added to generate true unique point IDs
from qdrant_client.http import models as rest_models
from qdrant_client import QdrantClient

# Add project root to imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DB_TYPE, QDRANT_URL, EMBED_MODEL, VECTOR_SIZE, COLLECTION_NAME
from src.langsmith_helper import get_traceable  # Import your active LangSmith helper

traceable = get_traceable()
VECTOR_SIZE = 384  # int for MiniLM vectors

_client = None

def get_db_client():
    global _client
    if _client is None:
        if VECTOR_DB_TYPE == "qdrant":
            print("Running Qdrant in Local Embedded Mode...")
            _client = QdrantClient(path="./qdrant_db")
        else:
            pass
            
    return _client

@traceable(name="Qdrant Add Chunks")
def add_chunks(chunks: list[dict], embeddings: list[list[float]]):
    client = get_db_client()
    
    collections_response = client.get_collections()
    existing_collections = [col.name for col in collections_response.collections]
    
    if COLLECTION_NAME not in existing_collections:
        print(f" Creating fresh Qdrant Collection: '{COLLECTION_NAME}' (384 Dimensions)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest_models.VectorParams(
                size=384,
                distance=rest_models.Distance.COSINE
            )
        )

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Generate a true random UUID string to prevent collision or trailing index leakage
        point_id = str(uuid.uuid4())
        
        points.append(
            rest_models.PointStruct(
                id=point_id, 
                vector=embedding, 
                payload={"text": chunk["text"], "page_num": chunk["page_num"]}
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Successfully synchronized and stored {len(chunks)} chunks in database.")

@traceable(name="Qdrant Query Chunks")
def query_chunks(query_embedding: list[float], top_k: int = 7) -> list[dict]:
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
            "score": round(point.score, 4)
        })

    return matched_chunks

@traceable(name="Qdrant Clear Collection")
def clear_collection():
    """Delete the active collection completely and confirm deletion."""
    client = get_db_client()
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        
        # Defensive check to block the engine until disk operations finish
        import time
        attempts = 0
        while client.collection_exists(collection_name=COLLECTION_NAME) and attempts < 10:
            time.sleep(0.1)
            attempts += 1
            
        print(f"Collection '{COLLECTION_NAME}' cleared successfully.")


if __name__ == "__main__":
    from embedder import embed_texts, embed_query

    test_chunks = [
        {"chunk_id": 0, "page_num": 1, "text": "Revenue grew 12% year over year."},
        {"chunk_id": 1, "page_num": 1, "text": "Operating expenses increased by 5%."},
        {"chunk_id": 2, "page_num": 2, "text": "The company faces regulatory risk in Europe."},
    ]
    
    print("--- Testing Integrated Qdrant Vector Engine Pipeline ---")
    clear_collection()  # Ensure cleanup run first
    embeddings = embed_texts([c["text"] for c in test_chunks])
    add_chunks(test_chunks, embeddings)

    q_emb = embed_query("What was the revenue growth?")
    results = query_chunks(q_emb, top_k=2)
    
    print("\n Matching Results Pulled:")
    for r in results:
        print(r)
        
    if _client:
        _client.close()