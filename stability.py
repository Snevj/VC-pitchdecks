"""
test_evaluator_stability.py
Runs ONE fixed question/answer/context through the Faithfulness judge
multiple times to check if low scores are real or evaluator noise.

Usage: place this in your project root (next to run_evals.py) and run:
    python test_evaluator_stability.py
"""

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory

from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations

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

faithfulness_metric = Faithfulness(llm=evaluator_llm)

# Fixed inputs — using the coffee/cocoa question + its known answer/context
# so we test ONLY evaluator variance, not retrieval/generation variance.
QUESTION = "What is the company's outlook regarding coffee and cocoa prices?"

# Run retrieval + generation ONCE, then re-score the SAME answer N times
async def get_fixed_answer():
    chunks = retrieve_and_rerank(QUESTION, top_k_retrieve=15, top_k_final=3)
    result = generate_with_citations(QUESTION, chunks)
    contexts = [c["text"] for c in chunks]
    return result["answer"], contexts


async def score_once(answer: str, contexts: list, delay: int) -> float:
    await asyncio.sleep(delay)
    result = await faithfulness_metric.ascore(
        user_input=QUESTION,
        response=answer,
        retrieved_contexts=contexts
    )
    return result.value, getattr(result, "reason", "")


async def main():
    print(f"Question: {QUESTION}\n")
    print("Running retrieval + generation once to get a fixed answer/context pair...")
    answer, contexts = await get_fixed_answer()

    print(f"\nAnswer:\n{answer}\n")
    print(f"Contexts ({len(contexts)} chunks):")
    for i, c in enumerate(contexts):
        print(f"  [{i+1}] {c[:150]}...")

    N_RUNS = 5
    DELAY = 15  # seconds between calls, same rate-limit-safe pattern as run_evals.py

    print(f"\nScoring the SAME answer {N_RUNS} times to check evaluator stability...")
    print(f"(~{N_RUNS * DELAY // 60}-{N_RUNS * DELAY // 60 + 1} minutes)\n")

    scores = []
    for i in range(N_RUNS):
        print(f"  [{i+1}/{N_RUNS}] Scoring...")
        score, reason = await score_once(answer, contexts, delay=DELAY)
        scores.append(score)
        print(f"    -> F={score:.3f}" + (f" | Reason: {reason}" if reason else ""))

    print("\n" + "─" * 50)
    print("STABILITY RESULTS")
    print("─" * 50)
    print(f"Scores: {[round(s, 3) for s in scores]}")
    print(f"Mean:   {sum(scores)/len(scores):.3f}")
    print(f"Min:    {min(scores):.3f}")
    print(f"Max:    {max(scores):.3f}")
    print(f"Range:  {max(scores) - min(scores):.3f}")

    spread = max(scores) - min(scores)
    if spread > 0.3:
        print("\n⚠️  HIGH VARIANCE — this looks like evaluator noise, not a real pipeline issue.")
        print("   The 8B judge model is likely unreliable for this question/context pair.")
    elif spread > 0.1:
        print("\n🟡 MODERATE VARIANCE — some inconsistency, worth a second opinion (e.g. larger judge model).")
    else:
        print("\n🟢 LOW VARIANCE — the low score appears to be a real, consistent judgment.")
        print("   Worth investigating the answer/context pair itself rather than evaluator reliability.")


if __name__ == "__main__":
    asyncio.run(main())