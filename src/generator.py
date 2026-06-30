"""
generator.py
Builds a prompt from retrieved chunks and calls high-speed Groq inference.
Optimized for ultra-low latency, dynamic citations, and local disk caching.
"""

import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Project tracing
from langsmith_helper import get_traceable
traceable = get_traceable()

@traceable(name="generate_with_citations")
def generate_with_citations(query: str, retrieved_chunks: list) -> dict:
    context_text = "\n\n".join([
        f"[Source Page: {chunk.get('page_num', 'Unknown')}]:\n{chunk.get('text', '')}"
        for chunk in retrieved_chunks
    ])
    
    system_instruction = (
        "You are an elite financial analyst. Answer the user's question using ONLY the provided context blocks.\n"
        "Carefully read the ENTIRE context before answering — relevant information may appear anywhere in the "
        "text, including under specific headers (e.g. 'Commodity Outlook', 'Financial Highlights'). Prioritize "
        "content that directly and explicitly addresses the question over generally related but tangential content.\n"
        "Never state a fact, figure, trend, or claim that is not explicitly present in the context blocks above, "
        "even if it seems plausible or commonly known. If you are unsure whether a claim is grounded, omit it.\n\n"
        "Your output MUST be a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "answer": "Your detailed answer string here",\n'
        '  "pages_referenced": [20, 21]\n'
        "}\n"
        "Only include page integers in the array if you extracted explicit facts from that block.\n\n"
        f"Context:\n{context_text}"
    )
    
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": query}
    ]
    
    print("🧠 Invoking model via native low-latency JSON mode...")
    response = llm.invoke(messages)
    
    try:
        parsed = json.loads(response.content)
        return {
            "answer": parsed["answer"],
            "pages_referenced": sorted(list(set(parsed["pages_referenced"]))),
            "tokens_used": len(response.content.split())
        }
    except Exception:
        return {
            "answer": response.content,
            "pages_referenced": sorted(list(set([c.get('page_num', 0) for c in retrieved_chunks]))),
            "tokens_used": 0
        }


# ── ⚡ ADD CACHING LAYER INFRASTRUCTURE HERE ──────────────────────────────

def get_cache_key(query: str) -> str:
    """Generate a unique MD5 hash signature for the input query string."""
    return hashlib.md5(query.encode()).hexdigest()


@traceable(name="generate_with_cache")
def generate_with_cache(query: str, chunks: list) -> dict:
    """Check local file storage for a pre-computed answer before hitting the API."""
    cache_dir = ".cache"
    cache_file = f"{cache_dir}/{get_cache_key(query)}.json"
    
    # 1. Look for pre-existing execution footprints
    if os.path.exists(cache_file):
        print("⚡ Cache hit! Returning saved response parameters instantly.")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    # 2. Cache Miss: Execute the active LLM sequence
    print("❌ Cache miss. Routing request to inference engine...")
    result = generate_with_citations(query, chunks)
    
    # 3. Synchronize output structure to disk memory
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)
        
    return result


# ── 🔄 UPDATE THE PUBLIC WRAPPER TO ROUTE THROUGH THE CACHE ──────────────

@traceable(name="generate")
def generate(query: str, chunks: list[dict]) -> str:
    """Return a plain answer string by routing through the local cache layer."""
    # Swapped from generate_with_citations to generate_with_cache
    result = generate_with_cache(query, chunks)
    
    if isinstance(result, dict):
        return result.get("answer", "")
    return str(result)