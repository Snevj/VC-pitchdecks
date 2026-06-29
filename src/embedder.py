import os
import sys
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
#this below line will prompt the python file to look inside the root folder for the config.py file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBED_MODEL

# Project tracing
from langsmith_helper import get_traceable
traceable = get_traceable()

_model = None
#loading the modal only once and storing it in a global variable to avoid reloading it multiple times
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model
#These functions isolate the raw embedding step. They grab the initialized model and call .encode()
#also converts into a python compatible list of floats for easier handling and storage
@traceable(name="embed_texts")
def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

@traceable(name="embed_query")
def embed_query(query: str) -> list[float]:
    model = get_model()
    return model.encode(query).tolist()


if __name__ == "__main__":
    print("--- Testing Native Similarity ---")
    model = get_model()
    
    documents = [
        "Revenue grew 12% year over year.",
        "Net income was $4.2 billion.",
        "The team went out for a casual pizza lunch on Friday."
    ]
    query = ["What was the revenue growth?"]
    
    # Generate embeddings using my pipeline functions
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode(query)
    
    # Using SentenceTransformer's native similarity engine directly for cosine similarity
    similarity_scores = model.similarity(query_embedding, doc_embeddings)
    
    print(f"\n Query: '{query[0]}'\n")
    # Extract the matching array scores from the PyTorch Tensor output
    for i, score in enumerate(similarity_scores[0]):
        print(f"Doc [{i+1}]: '{documents[i]}'")
        print(f"Similarity Score: {score.item():.4f}")
        print("-" * 40)