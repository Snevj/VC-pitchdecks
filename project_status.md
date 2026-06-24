## Status: 

'''
Current:-
Next:-
'''

## Implementation
### starting with the architecture of the whole project

<!-- financial-rag/
│
├── data/                    # drop your PDFs here
│
├── src/
│   ├── loader.py            # PDF text extraction
│   ├── chunker.py           # split text into chunks
│   ├── embedder.py          # sentence-transformers embedding
│   ├── vectorstore.py       # ChromaDB store + query
│   ├── retriever.py         # search + rerank
│   ├── generator.py         # Ollama call + prompt builder
│   └── pipeline.py          # connects all steps end to end
│
├── evals/
│   └── test_questions.json  # your 20 hand-written Q&A pairs
│
├── app.py                   # Gradio UI (Week 4 only)
├── requirements.txt
└── .env                     # API keys (Cohere, LangSmith) -->
[x]made the complete structure
### now moving to making the basic RAG + PDF chunking process with the help of Vector DB