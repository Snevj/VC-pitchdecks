# FinRAG — Financial Document Intelligence System

> A containerized, production-grade RAG system for querying financial documents with exact source citations. Deployed on AWS EC2 via Docker with end-to-end LangSmith observability and RAGAS v0.4 evaluation.

---

## What It Does

Upload any financial PDF (earnings reports, annual filings, quarterly results) and ask natural language questions. FinRAG retrieves the most relevant passages, reranks them for precision, and generates a grounded answer with exact page citations — no hallucination, no guessing.

```
Upload PDF → Extract → Chunk → Embed → Store in Qdrant
                                              ↓
Ask question → Vector Search (top-15) → Cohere Rerank (top-3) → Groq LLM → Answer + Page Citations
```

---

## Evaluation Results

Scored against hand-written ground truth Q&A pairs using **RAGAS v0.4** with `llama-3.1-8b-instant` as the evaluator LLM via Groq.

| Metric | Score | What It Measures |
|---|---|---|
| **Context Precision** | **0.97** | Are the retrieved chunks actually relevant to the query? |
| **Answer Relevancy** | **0.88** | Does the answer address the question asked? |
| **Faithfulness** | **0.77** | Are answer claims grounded in the retrieved context? |

**Latency** (traced via LangSmith):
| Component | p50 Latency |
|---|---|
| `retrieve_and_rerank` | 1.2 – 1.5s |
| `generate_with_citations` | 0.73 – 1.57s |
| End-to-end | ~2 – 3s per query |

---

## Architecture

| Layer | Component | Purpose |
|---|---|---|
| API | FastAPI + Uvicorn | Async REST backend |
| Document Parsing | PyMuPDF | Text + page number extraction |
| Chunking | Recursive character splitter | Sentence-boundary aware chunks (~500 chars) |
| Embeddings | `all-MiniLM-L6-v2` (local) | 384-dim semantic vectors, zero API cost |
| Vector Store | Qdrant (local embedded mode) | Cosine similarity search, no server required |
| Reranking | Cohere Rerank API | Cross-encoder precision: top-15 → top-3 |
| Generation | Groq (`openai/gpt-oss-120b`) | Sub-second LLM inference via LPU hardware |
| Structured Output | Pydantic + JSON mode | Enforces `answer` + `pages_referenced` schema |
| Evaluation | RAGAS v0.4 | Faithfulness, Answer Relevancy, Context Precision |
| Observability | LangSmith (`@traceable`) | Full pipeline latency + I/O tracing |
| Deployment | Docker → AWS EC2 | Containerized, cross-compiled for linux/amd64 |

---

## Project Structure

```
finrag/
├── src/
│   ├── loader.py            # PDF text + page number extraction (PyMuPDF)
│   ├── chunker.py           # Recursive character text splitter
│   ├── embedder.py          # Local sentence-transformers pipeline
│   ├── vectorstore.py       # Qdrant index creation + ingestion
│   ├── retriever.py         # Vector search (top-15) + Cohere rerank (top-3)
│   ├── generator.py         # Prompt builder + Groq structured output
│   ├── pipeline.py          # End-to-end async orchestration
│   └── langsmith_helper.py  # Central @traceable tracing utility
├── evals/
│   └── test_questions.json  # Hand-written ground truth Q&A pairs
├── experiments/
│   └── finrag_v0.4_results.csv  # RAGAS evaluation output
├── app.py                   # FastAPI backend
├── run_evals.py             # RAGAS v0.4 async evaluation runner
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Free API keys: [Cohere](https://cohere.com), [Groq](https://console.groq.com), [LangSmith](https://smith.langchain.com)

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
python app.py
# Visit http://localhost:8000
```

---

## Evaluation

RAGAS v0.4 scores computed against hand-written ground truth Q&A pairs.

```bash
python run_evals.py
```

**Implementation notes:**
- Uses `AsyncOpenAI` client pointed at Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`)
- Metrics instantiated via `llm_factory()` which returns `InstructorLLM` — the only type accepted by RAGAS v0.4 collections metrics
- Evaluation runs sequentially (not concurrently) with inter-metric delays to respect Groq's free tier TPM limits (6,000 tokens/minute)
- `ascore()` called directly on each metric — bypasses the deprecated `evaluate()` / `aevaluate()` functions which have a class hierarchy mismatch with v0.4 collection metrics

```python
# v0.4 correct pattern
f_result  = await faithfulness_metric.ascore(user_input=q, response=a, retrieved_contexts=ctx)
ar_result = await answer_relevancy_metric.ascore(user_input=q, response=a)
cp_result = await context_precision_metric.ascore(user_input=q, reference=ref, retrieved_contexts=ctx)
```

Results are saved to `experiments/finrag_v0.4_results.csv` with per-question scores and LLM-generated reasoning.

---

## Docker Deployment

```bash
# Build — cross-compile for AWS EC2 (x86_64) from Apple Silicon (ARM)
docker buildx build --platform linux/amd64 -t snehvj/finrag .

# Push to Docker Hub
docker push snehvj/finrag

# On AWS EC2
docker pull snehvj/finrag
docker run -d -p 80:8000 --env-file .env snehvj/finrag
```

### AWS EC2 Setup

- Instance: `t3.micro` (Ubuntu 22.04)
- Storage: 20 GiB EBS (expanded from 8 GiB default — ML dependencies + Qdrant data exceed the default)
- Security Group: Port `80` open to `0.0.0.0/0`

#### EBS Storage Expansion

After resizing the EBS volume in AWS Console, expand the partition and filesystem without downtime:

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/root
```

---

## Key Engineering Decisions

**Why Qdrant over ChromaDB?**
Qdrant's local embedded mode is faster for single-collection use cases. The SDK gives direct control over distance metrics and payload filtering — useful when scaling to multi-document retrieval with metadata filtering by date, company, or filing type.

**Why a two-stage retrieval pipeline?**
Vector search (Stage 1) maximises recall — it casts a wide net across all chunks using embedding similarity. Cohere's cross-encoder reranker (Stage 2) reads the query and each chunk *together*, dramatically improving precision. For financial questions where exact phrasing matters ("net profit" vs "profit after exceptional items"), this combination outperforms pure vector search significantly. Result: top-15 → top-3 with Context Precision of 0.97.

**Why Groq for inference?**
Groq's LPU (Language Processing Unit) hardware delivers 10–30x faster inference than standard GPU-hosted APIs for single-request latency. Generation p50 is 0.85s — critical for a responsive chat interface. The OpenAI-compatible API means zero code changes to swap models.

**Why structured output over freeform generation?**
Pydantic schema enforcement via JSON mode guarantees the LLM always returns `{"answer": "...", "pages_referenced": [13, 22]}`. No string parsing, no citation extraction heuristics, no silent failures. If the schema is violated, a fallback chain catches it.

**Why call `ascore()` directly instead of `evaluate()`?**
RAGAS v0.4's `evaluate()` function validates metrics using the old `Metric` base class, but v0.4 collection metrics inherit from `BaseMetric` — a different class hierarchy. This causes a `TypeError` regardless of how metrics are instantiated. Calling `ascore()` directly bypasses this internal validator entirely and is the correct v0.4 pattern.

---

## Tech Stack

`Python` `FastAPI` `Qdrant` `sentence-transformers` `Cohere Rerank` `Groq` `LangChain` `RAGAS v0.4` `LangSmith` `Docker` `AWS EC2` `PyMuPDF` `Pydantic` `asyncio`

---

## Author

**Sneh Vijayvargiya**  
[GitHub](https://github.com/Snevj)