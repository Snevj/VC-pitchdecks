import os
import json
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
# ── 🛠️ IMPORT LANGCHAIN GROQ WRAPPERS ─────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import your working pipeline modules
from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations

def evaluate_pipeline():
    print("🧪 Loading test evaluation dataset...")
    with open("evals/test_questions.json", "r") as f:
        test_cases = json.load(f)

    evaluation_rows = []

    print(f"🏃‍♂️ Running {len(test_cases)} questions through your local pipeline...")
    for case in test_cases:
        query = case["question"]
        chunks = retrieve_and_rerank(query, top_k_retrieve=15, top_k_final=3)
        result = generate_with_citations(query, chunks)
        
        evaluation_rows.append({
            "question": query,
            "answer": result["answer"],
            "contexts": [c["text"] for c in chunks],
            "ground_truth": case["answer"]
        })

    dataset = Dataset.from_list(evaluation_rows)
    
    # ── SET UP THE CLOUD EVALUATOR JUDGES ─────────────────────────────────────
    evaluator_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📊 Calculating RAGAS metrics in the cloud via Groq...")
    scores = evaluate(
        dataset=dataset,
        # 🛠️ FIX: Don't pass llm= into the metrics here. Let them inherit it!
        metrics=[
            Faithfulness(), 
            AnswerRelevancy(), 
            ContextPrecision()
        ],
        llm=evaluator_llm,             # RAGAS will auto-convert this to its InstructorLLM under the hood!
        embeddings=evaluator_embeddings 
    )
    
    print("\n🏆 🎉 ── FINAL EVALUATION SCORES ──")
    print(json.dumps(scores, indent=4))
    return scores

if __name__ == "__main__":
    evaluate_pipeline()