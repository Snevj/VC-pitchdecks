import os
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# Database Selection Config 
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "./qdrant_db")

# RAG Configurations
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile") 
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Directory Structures
QDRANT_DIR = "./qdrant_db"   # Swapped from Chroma to Qdrant!
UPLOAD_DIR = "./uploads"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_docs")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))