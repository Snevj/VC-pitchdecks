import os
import json
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from openai import OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations


def evaluate_pipeline():
    print("Loading test evaluation dataset...")
    with open("evals/test_questions.json", "r") as f:
        test_cases = json.load(f)

    evaluation_rows = []

    print(f"Running {len(test_cases)} questions through pipeline...")
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
    
    # ── 🛠️ INITIALIZE CHAT AND EMBEDDING ENGINES ─────────────────────────────
    raw_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    raw_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ── 🛠️ WRAP THEM FOR EXPLICIT RAGAS COMPATIBILITY ────────────────────────
    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    print("Calculating RAGAS metrics in the cloud via Groq...")
    scores = evaluate(
        dataset=dataset,
        # 🛠️ PASS THE PROPERLY WRAPPED INSTANCES INTO EACH CONSTRUCTOR INDIVIDUALLY
        metrics=[
            Faithfulness(llm=evaluator_llm), 
            AnswerRelevancy(llm=evaluator_llm), 
            ContextPrecision(llm=evaluator_llm)
        ],
        llm=evaluator_llm,             
        embeddings=evaluator_embeddings 
    )
    
    print("\nFINAL EVALUATION SCORES")
    print(json.dumps(scores, indent=4))
    return scores


if __name__ == "__main__":
    evaluate_pipeline()
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load configuration and API keys
load_dotenv()

# Import your working local pipeline blocks
from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations

# Initialize the cloud judge directly via the official Groq client
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in your .env file!")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_groq_judge(system_prompt: str, user_prompt: str) -> float:
    #Helper to get a clean mathematical score from our Groq cloud judge.
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Zero out creativity for strict mathematical grading
            max_tokens=10
        )
        output = response.choices[0].message.content.strip()
        # Extract the first float number found in the text response
        for word in output.replace("\n", " ").split():
            clean_word = "".join(c for c in word if c.isdigit() or c == ".")
            if clean_word:
                return min(1.0, max(0.0, float(clean_word)))
        return 0.0
    except Exception as e:
        print(f"Judging skip due to API rate limit: {e}")
        return 0.0

def run_custom_evaluations():
    print("Loading test evaluation dataset...")
    with open("evals/test_questions.json", "r") as f:
        test_cases = json.load(f)

    print(f"Processing {len(test_cases)} evaluation cases through the live local RAG index...")
    
    total_faithfulness = 0.0
    total_relevancy = 0.0
    results_table = []

    for i, case in enumerate(test_cases):
        query = case["question"]
        ground_truth = case["answer"]
        
        # 1. Run your actual working pipeline chunks + answers
        chunks = retrieve_and_rerank(query, top_k_retrieve=15, top_k_final=3)
        result = generate_with_citations(query, chunks)
        
        generated_answer = result["answer"]
        retrieved_context = "\n\n".join([c["text"] for c in chunks])

        # 2. Evaluate Faithfulness (Is it hallucinating?)
        faith_system = (
            "You are an expert financial auditor. Your task is to rate the FAITHFULNESS of a given answer based "
            "strictly on the provided context text. If the answer contains facts or numbers not found anywhere in the context, "
            "or contradicts the context, score it close to 0.0. If every claim in the answer is completely supported by the context, "
            "score it a perfect 1.0. Output ONLY a single floating-point number between 0.0 and 1.0."
        )
        faith_user = f"Context:\n{retrieved_context}\n\nGenerated Answer:\n{generated_answer}"
        faith_score = ask_groq_judge(faith_system, faith_user)

        # 3. Evaluate Answer Relevancy (Did it actually answer the query directly?)
        relevancy_system = (
            "You are an expert communications analyst. Rate the RELEVANCY of the generated answer to the original user question. "
            "Does it answer the question directly, concisely, and cleanly? If it talks about unrelated data or avoids the point, score it close to 0.0. "
            "If it answers the question precisely, score it 1.0. Output ONLY a single floating-point number between 0.0 and 1.0."
        )
        relevancy_user = f"Question:\n{query}\n\nGenerated Answer:\n{generated_answer}"
        relevancy_score = ask_groq_judge(relevancy_system, relevancy_user)

        total_faithfulness += faith_score
        total_relevancy += relevancy_score

        print(f"  [{i+1}/{len(test_cases)}] Faithfulness: {faith_score:.2f} | Relevancy: {relevancy_score:.2f}")
        
        results_table.append({
            "case": i + 1,
            "question": query,
            "faithfulness": faith_score,
            "relevancy": relevancy_score
        })

    # Calculate average metrics
    avg_faith = total_faithfulness / len(test_cases)
    avg_rel = total_relevancy / len(test_cases)

    print("\n ── FINAL EVALUATION SCORES ──")
    print(f"==========================================")
    print(f"Average Pipeline Faithfulness:  {avg_faith:.4f}")
    print(f"Average Pipeline Relevancy:     {avg_rel:.4f}")
    print(f"==========================================")


if __name__ == "__main__":
    # Ensure the official groq client package is available
    try:
        import groq
    except ImportError:
        os.system("pip install groq")
        
    run_custom_evaluations()
"""