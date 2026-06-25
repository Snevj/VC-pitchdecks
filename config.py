import os
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# Database Selection Config 
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "./qdrant_db")

# RAG Configurations
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Directory Structures
CHROMA_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
MODEL_CHUNKER = os.getenv("MODEL_CHUNKER", "llama3.2:3b")