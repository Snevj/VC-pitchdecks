"""
pipeline.py
Connects all steps end to end.
Week 1: index a PDF, then query it
Week 2: swap in recursive chunker + reranker
Week 3: add RAGAS evals + LangSmith tracing
Week 4: LangGraph state graph
"""

import os
import sys
from dotenv import load_dotenv

# Ensure configuration constants at root can be imported cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from loader import load_pdf
from chunker import chunk_text
from embedder import embed_texts
from vectorstore import add_chunks, clear_collection
from retriever import retrieve, retrieve_and_rerank
from generator import generate, generate_with_citations


# ── WEEK 1: Baseline MVP (Index + Query) ──────────────────────────────────────
# [Raw PDF] ──► load_pdf() ──► chunk_text() ──► embed_texts() ──► add_chunks()
def index_pdf_week1(pdf_path: str):
    """Load PDF → naive chunks → embed → store in database index."""
    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    add_chunks(chunks, embeddings)
    print(f"\nIndexed {pdf_path} successfully.")


# [User Question] ──► retrieve() ──► generate() ──► Final Answer
def query_week1(question: str) -> str:
    """Retrieve top 5 chunks → generate answer using base engine config."""
    # Corrected: simply call your high-level retrieve module directly using the string question
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
    # Corrected: utilize your custom internal retrieve_and_rerank directly
    chunks = retrieve_and_rerank(question, top_k_retrieve=15, top_k_final=3)
    result = generate_with_citations(question, chunks)
    
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: pages {result['sources']}")
    print(f"Tokens used: {result['token_count']}")
    return result


# ── WEEK 3: Evals + LangSmith Tracing ─────────────────────────────────────────
def query_week3(question: str) -> dict:
    """
    Same as Week 2 but wrapped inside a LangSmith tracing tree.
    """
    from langsmith import traceable

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
    """
    Loads verification test matrix, batches requests, and processes RAGAS scores.
    """
    import json
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    with open("evals/test_questions.json") as f:
        test_cases = json.load(f)

    rows = []
    for tc in test_cases:
        result = query_week3(tc["question"])
        rows.append({
            "question": tc["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": tc["answer"]  # Hand-crafted validation reference ground truth
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    print("\n── RAGAS Scores ──────────────────")
    print(scores)
    return scores


# ── WEEK 4: LangGraph State Graphs ────────────────────────────────────────────
def build_langgraph_pipeline():
    """
    Constructs a StateGraph state machine configuration pipeline complete with cyclic retry logic.
    """
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


# ── ENTRY POINT SYSTEM EXECUTION ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

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