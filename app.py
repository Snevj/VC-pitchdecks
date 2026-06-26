"""
app.py
Week 4: Gradio UI — upload a PDF, ask questions, see answers + citations + faithfulness score.
Run with: python app.py
"""

import gradio as gr
from rich.prompt import result
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
with gr.Blocks(title="Financial RAG", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Financial Document Q&A")
    gr.Markdown("Upload an earnings report or annual report PDF, then ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            index_btn = gr.Button("Index Document", variant="primary")
            index_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Your Question", placeholder="What was the revenue growth?", lines=2)
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)
            sources_output = gr.Textbox(label="Sources", lines=2, interactive=False)

    index_btn.click(fn=index_document, inputs=[pdf_input], outputs=[index_status])
    ask_btn.click(fn=answer_question, inputs=[question_input], outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()