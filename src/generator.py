"""
generator.py
Builds a prompt from retrieved chunks and calls Ollama (free, local).
:-basic prompt + answer
:-adds source citations and token counting
"""

# import ollama
# import tiktoken
# import config
# from transformers import AutoTokenizer
# from langchain_groq import ChatGroq
# MODEL = config.LLM_MODEL

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def generate_with_citations(query: str, retrieved_chunks: list) -> dict:
    # 1. Compile context with the correct 'page_num' key mapping
    context_text = "\n\n".join([
        f"[Source {i+1}]: {chunk.get('text', '')} (Page {chunk.get('page_num', 'Unknown')})"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    # 2. Build system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite financial analyst. Answer the user's question using ONLY the provided context blocks. "
            "If the context does not contain the answer, say 'I cannot find that information in the document.'\n\n"
            f"Context:\n{context_text}"
        )),
        ("human", "{question}")
    ])
    
    # 3. Cloud LLM via Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
    
    chain = prompt | llm
    
    print(f"🧠 Generating answer using cloud Llama via Groq...")
    response = chain.invoke({"question": query})
    
    # 4. Extract and sort unique page numbers safely from 'page_num'
    raw_pages = []
    for chunk in retrieved_chunks:
        p = chunk.get('page_num')
        if p is not None:
            raw_pages.append(p)
            
    pages = sorted(list(set(raw_pages))) if raw_pages else ["?"]
    
    return {
        "answer": response.content,
        "pages_referenced": pages,
        "tokens_used": len(response.content.split())
    }




























"""
# basic generation(prototype only)
def generate(query: str, chunks: list[dict]) -> str:
    ""
    Builds a prompt from chunks + query, calls Ollama, returns answer.
    ""
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f""You are a financial analyst assistant.
    Answer the question using ONLY the context provided below.
    If the answer is not in the context, say "I don't have enough information to answer this."

    Context:
    {context}

    Question: {query}

    Answer:""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
                "temperature": 0.2,       # Low (0.1 - 0.3) = Strict & accurate for RAG. High (0.7+) = Creative.
                "num_predict": 256,       # Caps the maximum number of tokens generated in the output answer.
                "top_k": 20,              # Limits the token pool size the LLM samples from during generation.
                "top_p": 0.9              # Nucleus sampling probability threshold modifier.
        }
    )
    return response["message"]["content"]


# generation with citations + token count (what we will be using in the final app )
def generate_with_citations(query: str, chunks: list[dict]) -> dict:
    ""
    Same as generate() but:
    - Tells the model to cite page numbers
    - Counts tokens before sending (so you understand cost)
    - Returns answer + sources + token count
    ""
    # build numbered context with page references
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[Source {i+1}, Page {c['page_num']}]:\n{c['text']}")
    context = "\n\n".join(context_parts)

    prompt = f""You are a financial analyst assistant.
    Answer the question using ONLY the context provided.
    After your answer, list which sources you used (e.g. "Sources used: Source 1, Source 3").
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {context}

    Question: {query}

    Answer:""

    # count tokens before sending (Week 2 — token awareness)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(prompt))
        print(f"Prompt token count: {token_count}")
    except Exception:
        token_count = None

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # extract which sources were cited
    sources = [c["page_num"] for c in chunks]

    return {
        "answer": answer,
        "sources": sources,
        "token_count": token_count,
        "chunks_used": len(chunks)
    }


if __name__ == "__main__":
    # quick test with fake chunks
    fake_chunks = [
        {"text": "Revenue grew 12% year over year to $94.8 billion.", "page_num": 3},
        {"text": "Operating income increased by 8% driven by cost efficiencies.", "page_num": 3},
    ]

    print("--- Basic generation ---")
    answer = generate("What was the revenue growth?", fake_chunks)
    print(answer)

    print("\n--- With citations ---")
    result = generate_with_citations("What was the revenue growth?", fake_chunks)
    print(result["answer"])
    print(f"Token count: {result['token_count']}")"""