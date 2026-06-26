## Status: 
```
Current:-
Next:-
```

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
- [x] made the complete structure
### now moving to making the basic RAG + PDF chunking process with the help of Vector DB
Made the basic pipeline for the project using RAG

### moving to advanced RAG pipeline and PDF chunking
Making the document loader using `pymupdf` which is independent, then initiated the chunking process with `RecursiveCharacterTextSplitter` from langchain. This followed by embedding of the chunks using `SentenceTransformer` using Hugging Face and this followed by a vector storage and this I kind of kept optional between `ChromaDB` and `Qdrant`, depending upon the use case, for local env and testing ChromaDB is good, but for scale Qdrant works better. And now comes the retrievel part, for this I used the embedding i used in the embedder and the `cohere` class for reranking the chunks retrieved. Now comes the generator file which uses llama3.2:3b model and `Ollama` and `tiktoken` for chat generation