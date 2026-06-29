"""
pipeline.py
Connects all steps end to end.
Week 1: index a PDF, then query it
Week 2: swap in recursive chunker + reranker
Week 3: add RAGAS evals + LangSmith tracing
Week 4: LangGraph state graph + Asynchronous Execution
"""

import os
import sys
import asyncio  # Added for high-concurrency async operations
from dotenv import load_dotenv

# Ensure configuration constants at root can be imported cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langsmith_helper import get_traceable
# Central traceable decorator for project-wide use (no-op if disabled)
traceable = get_traceable()

from loader import load_pdf
from chunker import chunk_text
from embedder import embed_texts
from vectorstore import add_chunks, clear_collection
from retriever import retrieve, retrieve_and_rerank
# Updated to track our structured, cached, and unified generator methods
from generator import generate, generate_with_citations, generate_with_cache 


# ── WEEK 1: Baseline MVP (Index + Query) ──────────────────────────────────────
def index_pdf_week1(pdf_path: str):
    """Load PDF → naive chunks → embed → store in database index."""
    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    add_chunks(chunks, embeddings)
    print(f"\nIndexed {pdf_path} successfully.")


def query_week1(question: str) -> str:
    """Retrieve top 5 chunks → generate answer using base engine config."""
    chunks = retrieve(question, top_k=5)
    answer = generate(question, chunks)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    return answer


# ── WEEK 2: Two-Stage Retrieval & Neural Reranking ────────────────────────────
def index_pdf_week2(pdf_path: str):
    """Load PDF → recursive chunks → embed → clean store migration."""
    clear_collection()  # Protect data hygiene by wiping old collection indices
    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    add_chunks(chunks, embeddings)
    print(f"\nIndexed {pdf_path} with recursive chunking structures.")


def query_week2(question: str) -> dict:
    """Retrieve 15 candidate chunks → Cohere rerank to 3 → generate answers with source tracking."""
    chunks = retrieve_and_rerank(question, top_k_retrieve=15, top_k_final=3)
    # Routed through generate_with_citations which now uses fast native JSON Mode mapping
    result = generate_with_citations(question, chunks)
    
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: pages {result['pages_referenced']}")
    print(f"Tokens used: {result['tokens_used']}")
    return result


# ── WEEK 3: Evals + LangSmith Tracing ─────────────────────────────────────────
def query_week3(question: str) -> dict:
    """Same as Week 2 but wrapped inside a LangSmith tracing tree."""
    @traceable(name="rag_pipeline")
    def _run(question):
        chunks = retrieve_and_rerank(question, top_k_retrieve=15, top_k_final=3)
        result = generate_with_citations(question, chunks)
        
        # Attach raw contexts so RAGAS can parse faithfulness thresholds mathematically
        result["contexts"] = [c["text"] for c in chunks]
        result["question"] = question
        return result

    return _run(question)


def run_evals():
    """Loads verification test matrix, batches requests, and processes RAGAS scores."""
    import json
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Set it in .env or export OPENAI_API_KEY=... to run evals.")
        return None

    with open("evals/test_questions.json") as f:
        test_cases = json.load(f)

    rows = []
    for tc in test_cases:
        result = query_week3(tc["question"])
        rows.append({
            "question": tc["question"],
            "answer": result.get("answer") if isinstance(result, dict) else None,
            "contexts": result.get("contexts") if isinstance(result, dict) else None,
            "ground_truth": tc["answer"]
        })

    dataset = Dataset.from_list(rows)

    try:
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    except Exception as e:
        err_msg = str(e)
        print("\nRAGAS evaluation failed:", err_msg)
        return None

    print("\n── RAGAS Scores ──────────────────")
    print(scores)
    return scores


# ── WEEK 4: LangGraph State Graphs ────────────────────────────────────────────
def build_langgraph_pipeline():
    """Constructs a StateGraph state machine configuration pipeline complete with cyclic retry logic."""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Optional

    class RAGState(TypedDict):
        question: str
        chunks: Optional[list]
        answer: Optional[str]
        contexts: Optional[list]
        faithfulness_score: Optional[float]
        retry_count: int

    def classify_node(state: RAGState) -> RAGState:
        print(f"[Node 1] Query: {state['question']}")
        return state

    def retrieve_node(state: RAGState) -> RAGState:
        chunks = retrieve_and_rerank(state["question"], top_k_retrieve=15, top_k_final=3)
        state["chunks"] = chunks
        state["contexts"] = [c["text"] for c in chunks]
        return state

    def generate_node(state: RAGState) -> RAGState:
        result = generate_with_citations(state["question"], state["chunks"])
        state["answer"] = result["answer"]
        return state

    def eval_node(state: RAGState) -> RAGState:
        from ragas.metrics import faithfulness
        from ragas import evaluate
        from datasets import Dataset

        dataset = Dataset.from_list([{
            "question": state["question"],
            "answer": state["answer"],
            "contexts": state["contexts"],
            "ground_truth": ""
        }])

        score = evaluate(dataset, metrics=[faithfulness])
        state["faithfulness_score"] = float(score["faithfulness"])
        print(f"[Node 4] Faithfulness score: {state['faithfulness_score']:.3f}")
        return state

    def should_retry(state: RAGState) -> str:
        if state["faithfulness_score"] < 0.7 and state["retry_count"] < 2:
            state["retry_count"] += 1
            print(f"[Gate] Score too low — initiating database retry loop branch {state['retry_count']}")
            return "retry"
        return "done"

    graph = StateGraph(RAGState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", eval_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_retry, {"retry": "retrieve", "done": END})

    return graph.compile()


def query_week4(question: str) -> dict:
    """Run execution state traversal over the compiled LangGraph orchestration model."""
    pipeline = build_langgraph_pipeline()
    result = pipeline.invoke({
        "question": question,
        "chunks": None,
        "answer": None,
        "contexts": None,
        "faithfulness_score": None,
        "retry_count": 0
    })
    print(f"\n {result['answer']}")
    print(f"Faithfulness: {result['faithfulness_score']}")
    return result


# ── PRODUCTION OPTIMIZATION: Async Multi-Threaded Execution Layer ────────────

@traceable(name="pipeline_async")
async def pipeline_async(query: str) -> dict:
    """
    Asynchronously executes retrieval and generation steps concurrently.
    Offloads heavy blocking CPU/Network tasks to a background worker pool.
    """
    print(f"🚀 Starting async pipeline worker loop for query: '{query}'")
    
    # 1. Concurrently handle Qdrant retrieval + reranking steps
    chunks = await asyncio.to_thread(retrieve_and_rerank, query)
    
    # 2. Concurrently handle generation via our local MD5 caching utility layer
    result = await asyncio.to_thread(generate_with_cache, query, chunks)
    
    return result


# ── ENTRY POINT SYSTEM EXECUTION ──────────────────────────────────────────────
if __name__ == "__main__":
    week = sys.argv[1] if len(sys.argv) > 1 else "1"
    pdf = "data/sample.pdf"

    if week == "1":
        index_pdf_week1(pdf)
        query_week1("What was the revenue growth?")

    elif week == "2":
        index_pdf_week2(pdf)
        query_week2("What are the main risk factors?")

    elif week == "3":
        run_evals()

    elif week == "4":
        query_week4("What was the net income for the year?")
        
    elif week.lower() == "async":
        # Interface to safely execute your new async loop via the CLI terminal
        print("\n--- Running High-Throughput Async Engine Mode ---")
        sample_q = "What are the four key priorities mentioned by the company as they look forward?"
        output_res = asyncio.run(pipeline_async(sample_q))
        print(f"\nAnswer: {output_res.get('answer')}")
        print(f"Pages Cited: {output_res.get('pages_referenced')}")