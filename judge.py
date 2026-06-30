"""
test_judge_with_full_chunk.py
Tests the Faithfulness judge using the EXACT, full, untrimmed chunk text
that the real pipeline retrieves — to check whether chunk noise/length
(unrelated boilerplate, signatures, multiple sub-topics in one chunk)
is what's degrading judge reliability, vs. the judge being inherently noisy.

Run this from your project root: python test_judge_with_full_chunk.py
"""

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")

client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
llm = llm_factory(model="llama-3.1-8b-instant", client=client)
metric = Faithfulness(llm=llm)

QUESTION = "What is the company's outlook regarding coffee and cocoa prices?"

ANSWER = (
    "The company expects coffee prices to keep falling, driven by a strong crop "
    "in Vietnam and an upcoming crop in Brazil. Cocoa prices are expected to stay "
    "subdued due to better supply and moderated demand."
)

# ── FULL chunk exactly as retrieved by the real pipeline (page 22) ──────────
FULL_CHUNK = """4/4 
• Powdered and Liquid Beverages delivered a strong and resilient performance, anchored by sustained 
double-digit growth in the coffee portfolio. A clearly defined strategy focused on penetration and 
premiumisation continued to build the category across both ends of the spectrum while driving 
sustainable value creation. Accelerated Ready-to-Drink journey with the launch of innovative Vietnamese 
Latte and Iced Cappuccino variants, reinforcing RTD as a key pillar of future growth. 
 
Key Highlights for the fourth quarter ended 31 March 2026: 
Total sales and domestic sales for the quarter increased by 23.4% and 23.1%, respectively. Domestic sales 
growth was broad based. Domestic sales crossed INR 6,445 crore. EBITDA margin stood at 26.3% 
 
 
Commodity Outlook:  
Coffee prices continue to trend lower, supported by a favourable crop in Vietnam and the forthcoming crop in 
Brazil. Cocoa prices remain subdued, reflecting improved supply and moderated demand. Sugar prices remain 
stable. Edible oil prices are firm and have moved higher in line with global crude oil prices, supported by increased 
diversion to biodiesel. Wheat has been affected by unseasonal rains in April, resulting in a delayed harvest and 
lower quantity and quality. Milk prices have firmed and are expected to remain elevated through the summer lean 
season. 
 
Cautionary Statement: 
Statements in this Press Release, particularly those which relate to outlook, describing the company's projections, 
estimates and expectations may constitute 'forward looking statements' within the meaning of applicable laws 
and regulations. Actual results might differ materially from those either expressed or implied in the statement 
depending on the circumstances. 
 
For more information 
Ambereen Ali Shah, ambereen.shah@in.nestle.com, +91 9717022731 
Amit Kumar Roy, amitkumar.roy@in.nestle.com, +91 8447737626 
Nestlé India Limited, Head Office: Nestlé House, Jacaranda Marg, M Block, DLF City Phase – II, Gurugram 122 002 (Haryana) 
Registered Office: 100 / 101, World Trade Centre, Barakhamba Lane, New Delhi – 110001,"""

TRIMMED_CHUNK = (
    "Commodity Outlook: Coffee prices continue to trend lower, supported by a "
    "favourable crop in Vietnam and the forthcoming crop in Brazil. Cocoa prices "
    "remain subdued, reflecting improved supply and moderated demand. Sugar prices "
    "remain stable."
)


async def score(label: str, context: str, delay: int = 5):
    await asyncio.sleep(delay)
    result = await metric.ascore(
        user_input=QUESTION,
        response=ANSWER,
        retrieved_contexts=[context]
    )
    print(f"{label}:")
    print(f"  Score: {result.value:.3f}")
    print(f"  Reason: {getattr(result, 'reason', 'NO REASON FIELD')}")
    print()
    return result.value


async def main():
    print(f"Question: {QUESTION}\n")
    print(f"Answer: {ANSWER}\n")
    print("=" * 60)
    print("TEST 1: Full, untrimmed chunk (exactly as pipeline retrieves it)")
    print("=" * 60)
    full_score = await score("Full chunk", FULL_CHUNK, delay=2)

    print("=" * 60)
    print("TEST 2: Trimmed chunk (only the relevant Commodity Outlook text)")
    print("=" * 60)
    trimmed_score = await score("Trimmed chunk", TRIMMED_CHUNK, delay=15)

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Full chunk score:    {full_score:.3f}")
    print(f"Trimmed chunk score: {trimmed_score:.3f}")
    diff = trimmed_score - full_score
    print(f"Difference:          {diff:+.3f}")

    if abs(diff) > 0.3:
        print("\n⚠️  LARGE DIFFERENCE — chunk noise/length appears to meaningfully")
        print("   degrade judge accuracy. Worth considering smaller, more focused")
        print("   chunks (splitting on section headers) to improve eval reliability.")
    else:
        print("\n🟢 SMALL DIFFERENCE — chunk length/noise is not the main driver.")
        print("   The judge itself is likely just inconsistent for this case;")
        print("   consider a stronger judge model (e.g. llama-3.3-70b-versatile)")
        print("   or averaging multiple judge runs.")


if __name__ == "__main__":
    asyncio.run(main())