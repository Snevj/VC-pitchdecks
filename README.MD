# FinRAG — Financial Document Intelligence System

> A containerized, production-grade RAG system for querying financial documents with exact source citations. Deployed on AWS EC2 via Docker.

---

## What it does

Upload any financial PDF (earnings reports, annual filings, quarterly results) and ask natural language questions. FinRAG retrieves the most relevant passages, reranks them for precision, and generates a grounded answer with exact page citations — no hallucination, no guessing.

```
Upload PDF → Extract → Chunk → Embed → Store in Qdrant
                                              ↓
Ask question → Vector Search (top 15) → Cohere Rerank (top 3) → Groq LLM → Answer + Page Citations
```

---

## Architecture

| Layer | Component | Purpose |
|---|---|---|
| API | FastAPI + Uvicorn | Async REST backend |
| Document Parsing | PyMuPDF | Text + page extraction |
| Chunking | Recursive character splitter | Sentence-boundary aware chunks |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, zero-latency semantic vectors |
| Vector Store | Qdrant (embedded) | Cosine similarity search |
| Reranking | Cohere Rerank API | Compress top-15 → top-3 chunks |
| Generation | Groq (Llama-3.3-70B) | Fast cloud LLM inference |
| Evaluation | RAGAS | Faithfulness + answer relevancy scoring |
| Observability | LangSmith | Full pipeline tracing |
| Deployment | Docker → AWS EC2 | Containerized cloud deployment |

---

## Project Structure

```
finrag/
├── src/
│   ├── loader.py          # PDF text extraction
│   ├── chunker.py         # Naive + recursive chunking
│   ├── embedder.py        # Local sentence-transformers
│   ├── vectorstore.py     # Qdrant store + query
│   ├── retriever.py       # Vector search + Cohere rerank
│   ├── generator.py       # Prompt builder + Groq LLM call
│   └── pipeline.py        # End-to-end orchestration (Week 1→4)
├── evals/
│   └── test_questions.json  # Ground truth Q&A pairs for RAGAS
├── app.py                 # FastAPI backend
├── run_evals.py           # RAGAS evaluation runner
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- API keys for Cohere (free), Groq (free), LangSmith (free)

### Installation

```bash
git clone https://github.com/Snevj/finrag.git
cd finrag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Fill in your keys
```

```env
COHERE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=finrag
```

### Run Locally

```bash
# Pull the LLM (first time only)
ollama pull llama3.2:3b

# Start the API server
python app.py

# Visit http://localhost:8000
```

---

## Evaluation

RAGAS scores are computed against hand-written ground truth Q&A pairs from the indexed document.

```bash
python run_evals.py
```

Metrics measured:
- **Faithfulness** — does the answer come from the retrieved context?
- **Answer Relevancy** — does the answer actually address the question?
- **Context Precision** — were the retrieved chunks relevant to the query?

---

## Docker Deployment

```bash
# Build (cross-compile for AWS EC2 x86_64 from Apple Silicon)
docker build --platform linux/amd64 -t snehvj/finrag .

# Push to Docker Hub
docker push snehvj/finrag

# On AWS EC2
docker pull snehvj/finrag
docker run -d -p 80:8000 --env-file .env snehvj/finrag
```

### AWS EC2 Setup

- Instance: `t3.micro` (Ubuntu)
- Storage: 20 GiB EBS (expanded from 8 GiB default to fit ML dependencies)
- Security Group: Port `80` open to `0.0.0.0/0`

#### Storage expansion (if needed)

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/root
```

---

## Key Engineering Decisions

**Why Qdrant over ChromaDB?**
Qdrant's local embedded mode is faster for single-collection use cases and the SDK gives more control over distance metrics and payload filtering — useful when scaling to multi-document retrieval with metadata.

**Why Cohere Rerank over pure vector search?**
Vector similarity finds semantically related chunks but doesn't understand query intent precisely. Cohere's cross-encoder reranker reads the query and each chunk together, dramatically improving precision for financial questions where exact phrasing matters.

**Why Groq for inference?**
Groq's LPU inference is significantly faster than standard GPU-hosted APIs for single-request latency — critical for a responsive chat interface. Free tier is sufficient for demo/portfolio use.

---

## Tech Stack

`Python` `FastAPI` `Qdrant` `sentence-transformers` `Cohere Rerank` `Groq` `RAGAS` `LangSmith` `Docker` `AWS EC2` `PyMuPDF`

---

## Author

**Sneh Vijayvargiya**
