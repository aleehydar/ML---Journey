# Pakistan Legal AI — Agentic RAG System

## What this project does
An intelligent legal assistant that answers questions about Pakistani law 
(Constitution 1973, Qanun-e-Shahadat 1984) and calculates income tax precisely.
Built with a production-grade Agentic RAG architecture.

## Architecture
- **Agentic Routing** — LLM autonomously decides which tool to call
- **HyDE Retrieval** — generates hypothetical legal documents to bridge 
  casual/formal language gap before searching
- **CrossEncoder Reranking** — ms-marco-MiniLM-L-6-v2 scores retrieved 
  chunks for relevance, filters anything below -5.0 threshold
- **FAISS Vector Store** — stores embeddings of Pakistani Constitution 
  and Qanun-e-Shahadat PDFs
- **SSE Streaming** — streams LLM tokens in real time to frontend
- **Hallucination Prevention** — model cannot answer outside its legal database

## Tools
| Tool | Purpose |
|------|---------|
| search_pakistan_law | HyDE + FAISS + CrossEncoder RAG pipeline |
| calculate_tax | Precise Pakistani tax slab calculator (2023-24) |

## Tech Stack
Python, FastAPI, LangChain, Groq (Llama 3.1), FAISS, 
sentence-transformers, CrossEncoder, Docker, Uvicorn

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Results
- Answers legal questions grounded strictly in Pakistani law
- Cites exact source document and page
- Tax calculations accurate to the rupee
- Real-time streaming responses