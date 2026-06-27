"""
app.py
Week 4: Gradio UI — upload a PDF, ask questions, see answers + citations + faithfulness score.
Run with: python app.py
"""
import os
import gradio as gr
from src.loader import load_pdf
from src.chunker import chunk_text  
from src.embedder import embed_texts
from src.vectorstore import add_chunks, clear_collection
from src.retriever import retrieve_and_rerank
from src.generator import generate_with_citations



def index_document(pdf_file) -> str:
    """Called when user uploads a PDF."""
    if pdf_file is None:
        return "No file uploaded."

    clear_collection()
    pages = load_pdf(pdf_file.name)
    chunks = chunk_text(pages)  
    embeddings = embed_texts([c["text"] for c in chunks])
    add_chunks(chunks, embeddings)

    return f"Indexed {len(chunks)} chunks from {len(pages)} pages. Ready to answer questions."



def answer_question(question: str) -> tuple[str, str]:
    """Called when user asks a question."""
    if not question.strip():
        return "Please enter a question.", ""

    # 1. Retrieve and rerank chunks using your retriever module
    reranked_chunks = retrieve_and_rerank(question, top_k_retrieve=15, top_k_final=3)
    
    # 2. Pass those exact reranked chunks into your updated Groq generator
    result = generate_with_citations(question, reranked_chunks)

    # 3. Format the metadata summary source block using Groq's new keys
    sources_summary = (
        f"Pages referenced: {', '.join(str(p) for p in result['pages_referenced'])}\n"
        f"Tokens (approx): {result['tokens_used']}"
    )

    return result["answer"], sources_summary

# ── UI layout ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile, uvicorn

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve your HTML file at localhost:8000/
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.post("/index")
async def index_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    clear_collection()
    pages = load_pdf(tmp_path)
    chunks = chunk_text(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    add_chunks(chunks, embeddings)

    return {"chunks": len(chunks), "pages": len(pages)}

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(body: Question):
    reranked_chunks = retrieve_and_rerank(body.question, top_k_retrieve=15, top_k_final=3)
    result = generate_with_citations(body.question, reranked_chunks)
    return {
        "answer": result["answer"],
        "pages_referenced": result["pages_referenced"],
        "tokens_used": result["tokens_used"]
    }

uvicorn.run(app, host="0.0.0.0", port=8000)