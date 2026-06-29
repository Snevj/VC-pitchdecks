"""
run_evals.py
Evaluates the FinRAG pipeline using RAGAS v0.4 ascore pattern.
Sequential execution with rate-limit-aware delays for Groq free tier.
"""

import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations

# ── 🛠️ GROQ ASYNC CLIENT ─────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")

groq_openai_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

evaluator_llm = llm_factory(
    model="llama-3.1-8b-instant",
    client=groq_openai_client
)
print(f"✅ Evaluator LLM ready: {type(evaluator_llm).__name__}")

# ── 🛠️ EMBEDDINGS ────────────────────────────────────────────────────────────
evaluator_embeddings = RagasHFEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# ── 🛠️ METRICS ───────────────────────────────────────────────────────────────
faithfulness_metric      = Faithfulness(llm=evaluator_llm)
answer_relevancy_metric  = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_precision_metric = ContextPrecision(llm=evaluator_llm)
print("✅ All metrics instantiated successfully")

# ── 🛠️ RATE-LIMIT-AWARE METRIC SCORER ────────────────────────────────────────
INTER_METRIC_DELAY = 15  # seconds between each metric call — stays under 6k TPM

async def score_with_retry(metric_fn, delay: int = INTER_METRIC_DELAY, **kwargs):
    """Call ascore with a pre-call delay to respect TPM limits."""
    await asyncio.sleep(delay)
    return await metric_fn(**kwargs)

# ── 🛠️ SINGLE ROW EVALUATOR ──────────────────────────────────────────────────
async def evaluate_row(tc: dict, idx: int, total: int) -> dict:
    query    = tc["question"]
    print(f"\n[{idx}/{total}] Evaluating: {query[:80]}...")

    chunks   = retrieve_and_rerank(query, top_k_retrieve=15, top_k_final=3)
    result   = generate_with_citations(query, chunks)
    answer   = result["answer"]
    contexts = [c["text"] for c in chunks]

    print(f"  ⏳ Scoring faithfulness...")
    f_result = await score_with_retry(
        faithfulness_metric.ascore,
        delay=INTER_METRIC_DELAY,
        user_input=query,
        response=answer,
        retrieved_contexts=contexts        # ✅ correct
    )

    print(f"  ⏳ Scoring answer relevancy...")
    ar_result = await score_with_retry(
        answer_relevancy_metric.ascore,
        delay=INTER_METRIC_DELAY,
        user_input=query,
        response=answer                    # ✅ no contexts — AnswerRelevancy doesn't take them
    )

    print(f"  ⏳ Scoring context precision...")
    cp_result = await score_with_retry(
        context_precision_metric.ascore,
        delay=INTER_METRIC_DELAY,
        user_input=query,
        reference=tc["answer"],            # ✅ reference, not response
        retrieved_contexts=contexts        # ✅ correct
    )

    print(f"  ✅ F={f_result.value:.3f} | AR={ar_result.value:.3f} | CP={cp_result.value:.3f}")

    return {
        "question":                 query,
        "answer":                   answer,
        "reference":                tc["answer"],
        "faithfulness":             f_result.value,
        "answer_relevancy":         ar_result.value,
        "context_precision":        cp_result.value,
        "faithfulness_reason":      getattr(f_result,  "reason", ""),
        "answer_relevancy_reason":  getattr(ar_result, "reason", ""),
        "context_precision_reason": getattr(cp_result, "reason", ""),
    }
# ── 🛠️ MAIN ──────────────────────────────────────────────────────────────────
async def main():
    print("\nLoading test evaluation dataset...")
    with open("evals/test_questions.json", "r") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    print(f"Evaluating {total} questions sequentially (rate-limit safe)...\n")
    print(f"⏱️  Estimated time: ~{total * 3 * INTER_METRIC_DELAY // 60}–{total * 3 * INTER_METRIC_DELAY // 60 + 2} minutes\n")

    # ── Sequential execution — prevents TPM burst ─────────────────────────────
    rows = []
    for idx, tc in enumerate(test_cases, start=1):
        row = await evaluate_row(tc, idx, total)
        rows.append(row)
        # Extra cooldown between questions (not just between metrics)
        if idx < total:
            print(f"  💤 Cooling down 10s before next question...")
            await asyncio.sleep(10)

    df = pd.DataFrame(rows)

    # ── Per-question results ──────────────────────────────────────────────────
    print("\n🏆 FINAL EVALUATION RESULTS")
    print("─" * 80)
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string(index=False))

    # ── Aggregate scores ──────────────────────────────────────────────────────
    print("\n📊 AGGREGATE SCORES")
    print("─" * 40)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        score = df[metric].mean()
        emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        print(f"  {emoji}  {metric:25s}: {score:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/finrag_v0.4_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Full results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
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