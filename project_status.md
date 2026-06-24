##Status: 

'''
Current:-
Next:-
'''

##Implementation
financial-rag/
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
└── .env                     # API keys (Cohere, LangSmith)