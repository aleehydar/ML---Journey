# Pakistan Legal AI — Agentic RAG System
## Complete Technical Architecture & Implementation

---

## Slide 1: Project Overview

### Pakistan Legal AI Assistant
**An intelligent legal question-answering system with precise tax calculation**

- **Purpose:** Answer Pakistani legal questions grounded in official documents (Constitution 1973, Qanun-e-Shahadat 1984, Penal Code, Family Laws, Corporate Laws, etc.)
- **Secondary Tool:** Precise Pakistani income tax calculator (2023-24 tax slabs)
- **Architecture:** Production-grade Agentic RAG (Retrieval-Augmented Generation)
- **Tech Stack:** Python, FastAPI, LangChain, LLM-based agent, FAISS vector database
- **Deployment:** Docker-ready, containerized with docker-compose

---

## Slide 2: System Workflow

### End-to-End Request Flow

```
User Question
    ↓
[FastAPI Endpoint] → JWT Auth Validation → Request Context Setup
    ↓
[Generation Service] → Initialize Agent + System Prompt
    ↓
Agent Decision Logic:
  ├─→ Call [Retrieval Tool] → HyDE + FAISS + BM25 + CrossEncoder Rerank
  ├─→ Call [Tax Calculator Tool] → Apply Tax Slabs
  └─→ Synthesize Answer with Citations
    ↓
Validation & Grounding Checks:
  ├─→ Verify sources cited
  ├─→ Check hallucination patterns
  └─→ Evaluate confidence
    ↓
[Stream Response via SSE] → Real-time tokens to frontend
    ↓
[Log & Evaluate] → Store metrics (faithfulness, relevance, recall)
    ↓
Response Complete
```

---

## Slide 3: Ingestion Pipeline

### Converting PDFs to Searchable Knowledge Base

**Entry Points:** `ingest.py`, `extract_and_load_pdfs.py`

**Pipeline:**
1. **PDF Discovery** → Find all `.pdf` files in `data/` folder
2. **Text Extraction** → PyMuPDF Loader or PyPDF2 extracts text from pages
3. **Normalization** → Clean whitespace, add OCR-aware formatting
4. **Provenance Metadata** → Attach source ID, checksum, effective date, parser version
5. **Recursive Chunking** → Split documents (chunk_size=500, overlap=100)
   - Splits at `\n\n` (paragraphs), then `\n` (lines), then `. ` (sentences)
   - Preserves legal context across chunk boundaries
6. **Embedding** → HuggingFace `all-MiniLM-L6-v2` model
7. **FAISS Index** → Persist vectorstore to disk at `vectorstore/`

**Output:** `legal_texts.json` (metadata) + `vectorstore/` (FAISS index)

---

## Slide 4: Storage Architecture

### Corpus & Persistence

**Primary Knowledge Store:**
- **File:** `legal_texts.json` — JSON array with document objects
- **Fields per item:** `source`, `text`, `organization_id` (tenant isolation)

**Vector Database:**
- **Technology:** FAISS (Facebook AI Similarity Search)
- **Location:** `vectorstore/` directory
- **Index Type:** Approximate nearest neighbor
- **Embedding Dimension:** 384 (all-MiniLM-L6-v2)
- **Capacity:** Can store thousands of document chunks

**Chunk Metadata:**
- `source` — filename + page number (e.g., "Constitution.pdf - Page 12")
- `checksum` — SHA256 of normalized text (integrity check)
- `effective_date` — ISO timestamp of ingestion
- `parser_version` — Track which OCR pipeline processed it
- `organization_id` — Multi-tenant isolation key

---

## Slide 5: Retrieval Service

### Hybrid Search & Intelligent Reranking

**Module:** `retrieval_service.py`

**Three-Stage Retrieval:**

1. **Intent Extraction**
   - LLM-based (or regex fallback) extracts filters: law_family, year, document_type
   - Example: "companies act 2017" → filter for company law docs from 2017

2. **Candidate Generation (Hybrid)**
   - **Dense Search:** FAISS similarity search (top 30 by embedding distance)
   - **Lexical Search:** BM25Okapi rank (or token-overlap fallback)
   - **Merge:** Deduplicated union of both result sets (30 candidates)
   - **Filter:** Apply org_id isolation + intent filters

3. **Reranking with CrossEncoder**
   - **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - **Input:** [query, document_text] pairs for all candidates
   - **Output:** Relevance scores for each pair
   - **Dynamic K:** Adaptive retrieval depth (select top-k based on score delta)
   - **Threshold:** Skip candidates scoring >2.0 below top candidate

**Return:** `RetrievalResult` with list of `RetrievalChunk` objects (source_id, text, score, metadata)

---

## Slide 6: Embedding & Vectorization

### Semantic Search Foundation

**Embedding Model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Dimension:** 384
- **Training:** MNLI + STS task finetuning
- **Speed:** Fast inference (CPU/GPU compatible)
- **Quality:** Good balance of quality vs. speed for legal domain

**Provider:** `embedding_provider.py`
- Forward-compatible imports (prefers `langchain_huggingface`, falls back to `langchain_community`)
- Automatically loads model from HuggingFace hub on first run

**Used By:**
- Ingestion pipeline: vectorizes document chunks → FAISS index
- Retrieval: queries transformed to embeddings for similarity search
- LangChain integration: seamless with LangChain vector store abstraction

---

## Slide 7: Generation Service & Agent

### LLM-Powered Tool-Calling Agent

**Module:** `generation_service.py`

**LLM Configuration:**
- **Provider:** Groq API (inference service)
- **Model:** `llama-3.1-8b-instant` (open-source, fast, legal-capable)
- **Library:** LangChain's `ChatGroq` wrapper

**Agent Setup:**
- **Tools Binding:** LLM can call 2 tools via native function calling
- **System Prompt:** Instructs model to ground answers in retrieved docs, cite sources
- **Message History:** Supports multi-turn context (previous Q&A)

**Streaming & Safety:**
- **SSE (Server-Sent Events):** Real-time token streaming to frontend
- **Sanitization:** Strips leaked tool placeholders from LLM output
- **Grounding Validation:** Checks answers cite [Source: X] for legal claims
- **Hallucination Prevention:** Rejects answers without proper grounding

**Evaluation Integration:**
- Post-generation runs RAGAS evaluator (faithfulness, answer relevance, context recall)
- Stores metrics in encrypted SQLite DB for later analysis

---

## Slide 8: Agent Tools

### Tool 1: Search Legal Database

**Function:** `search_legal_db_tool(query: str) -> str`

- **Input:** Natural language legal question
- **Process:**
  1. Calls `RetrievalService.retrieve(query, org_id=X, k=6)`
  2. Gets top 6 reranked chunks with source citations
  3. Returns formatted context blob with `[Source: ...]` tags
- **Output:** Formatted markdown with citations or "ERROR: INSUFFICIENT_EVIDENCE"
- **Used For:** All legal claims, constitutional questions, statutory interpretation

---

## Slide 9: Agent Tools (Continued)

### Tool 2: Calculate Tax

**Function:** `calculate_tax_tool(annual_income: float) -> str`

- **Input:** Annual income in Pakistani Rupees (PKR)
- **Tax Slabs (2023-24):**
  | Income Range | Rate |
  |---|---|
  | ≤ 600,000 | 0% |
  | 600K – 1.2M | 2.5% |
  | 1.2M – 2.4M | 12.5% |
  | 2.4M – 3.2M | 22.5% |
  | 3.2M – 4.1M | 27.5% |
  | > 4.1M | 35% |
- **Output:** Precise tax amount in PKR with source citation
- **Used For:** Tax queries, income-based legal consultations

---

## Slide 10: Request Context & Caching

### Request-Scoped Isolation & Performance

**Context Manager (`context_manager.py`):**
- **Per-Request Storage:** Unique `RequestContext` object per API call
- **Fields:** request_id, user_id, org_id, permissions
- **Mechanism:** Python `contextvars` for async safety
- **Scope Tracking:** Retrieved contexts stored separately for evaluation

**Semantic Cache (`cache/semantic_cache.py`):**
- **Purpose:** Avoid recomputing answers for identical queries
- **Key:** SHA256 hash of (org_id + query.lower().strip())
- **TTL:** 3600 seconds (1 hour) default
- **Lookup:** Returns cached result if not expired
- **Scalability:** In-memory dict; upgrade to Redis for multi-instance
- **Hit Rate:** Reduces LLM API costs on repeated queries

---

## Slide 11: Evaluation & Metrics

### Quality Assurance Framework

**Evaluator Modules (`evals/`):**
- **`generation_metrics.py`** — RAGAS evaluator for answer quality
  - Faithfulness: Is answer grounded in provided contexts?
  - Answer Relevance: Does answer address the question?
  - Context Recall: Are all relevant contexts retrieved?
- **`retrieval_metrics.py`** — Retrieval-specific metrics

**Metrics Computed per Query:**
- Faithfulness (0–1): confidence in grounding
- Answer Relevance (0–1): topical match to question
- Context Recall (0–1): coverage of retrieval
- Overall Score: average of three metrics
- Confidence Score: internal confidence threshold

**Thresholds:**
- `confidence_score < 0.7` → flag as requiring human review
- Missing sources → automatic abstention (ERROR: INSUFFICIENT_EVIDENCE)

---

## Slide 12: Database & Persistence

### Encrypted Evaluation Logging

**Module:** `db/schema.py`

**Storage:**
- **Engine:** SQLite 3 (`evaluations.db`)
- **Encryption:** Fernet (symmetric, key-based) from `cryptography` lib

**Table: `eval_logs`**
```
id, timestamp, user_id, org_id,
question_encrypted, answer_encrypted, contexts_encrypted,
faithfulness, answer_relevance, context_recall, overall_score
```

**Features:**
1. **PII Redaction** — Regex patterns mask CNIC, phone, email before storage
2. **Encryption** — Text fields encrypted with `ENCRYPTION_KEY` env var
3. **Tenant Isolation** — org_id enforced in all queries
4. **Audit Trail** — Timestamp all evaluations

**Query Methods:**
- Rolling averages by time period (7d, 30d, all-time)
- Time-series data for dashboards
- Delete by org or user (GDPR-friendly)
- Retention policy enforcement (delete after 90 days by default)

---

## Slide 13: Governance & Retention

### Data Lifecycle & Privacy

**Governance Endpoints (`api/routes/governance.py`):**
- `DELETE /governance/delete-by-org` — Remove all org's evaluation logs (requires `governance:write` permission)
- `DELETE /governance/delete-by-user/{user_id}` — Remove user's logs within org

**Retention Policy (`db/retention_policy.py`):**
- **Worker:** Background scheduler (runs daily at 02:00 UTC)
- **Default TTL:** 90 days (configurable)
- **Action:** Automatically deletes `eval_logs` older than TTL
- **Launch:** Separate process or scheduled task in production

**Permissions Model:**
- JWT claims include `org_id`, `sub` (user ID), `permissions` list
- Endpoints validate `require_permission("governance:write")` before deletion
- Prevents users from accessing/deleting other orgs' data

---

## Slide 14: Authentication & Authorization

### JWT-Based Multi-Tenant Security

**Module:** `auth_middleware.py`

**Auth Flow:**
1. Client sends request with `Authorization: Bearer <token>` header
2. Middleware extracts and validates JWT signature using `JWT_SECRET`
3. Decodes claims: `sub` (user), `org_id` (organization), `permissions` (list)
4. Creates `AuthClaims` object passed to route handlers

**Token Structure:**
```json
{
  "sub": "user@company.com",
  "org_id": "org-123",
  "permissions": ["chat:write", "governance:write"],
  "exp": 1234567890
}
```

**Permission Checks:**
- `require_permission("chat:write")` — Can use chat endpoint
- `require_permission("governance:write")` — Can delete logs
- Wildcard `"*"` — Admin access to all permissions

**Tenant Isolation:**
- Every query scoped to `org_id` from token
- Prevents cross-tenant data leakage

---

## Slide 15: API Endpoints

### FastAPI Routes

**Chat Endpoint** (`api/routes/chat.py`):
- **POST** `/api/v1/chat` (JSON response)
- **Request:** `QuestionRequest` (question, history, org_id)
- **Response:** `ChatResponse` (answer, sources, confidence_score, evaluation)
- **Auth:** Requires `chat:write` permission

**Streaming Chat Endpoint** (`generation_service.py`):
- **POST** `/api/v1/chat/stream` (SSE stream)
- **Format:** Server-Sent Events with `data:` lines (JSON objects)
- **Events:** trace (debug), token (answer chunks), sources, evaluation, done
- **Real-time:** Tokens streamed as LLM generates

**Governance Endpoints** (`api/routes/governance.py`):
- **DELETE** `/governance/delete-by-org` — Requires `governance:write`
- **DELETE** `/governance/delete-by-user/{user_id}` — Requires `governance:write`

**Infrastructure:**
- **Prometheus Metrics:** `GET /metrics` — Latency histograms, grounding counters, hallucination gauge
- **Swagger Docs:** `GET /docs` (FastAPI auto-generated)
- **ReDoc Docs:** `GET /redoc`

---

## Slide 16: Observability & Monitoring

### Distributed Tracing & Metrics

**Tracing (`monitoring/tracing.py`):**
- **Provider:** OpenTelemetry SDK
- **Exporter:** Console span exporter (dev) / upgradeable to Jaeger/Datadog
- **Spans:** Trace every HTTP request with method, URL, status code
- **Logging:** Standardized Python logging to stdout

**Metrics (`monitoring/metrics.py`):**
- **REQUEST_LATENCY** — Histogram of response times by method & endpoint
- **GROUNDING_PASS_COUNTER** — Count of answers that passed vs. abstained grounding checks
- **HALLUCINATION_RATE** — Gauge tracking rolling hallucination rate

**Middleware Integration** (`middleware/observability.py`):
- Wraps every FastAPI request with OpenTelemetry span
- Records start time, method, endpoint, status code, duration
- Observes latency metric

**Prometheus Integration:**
- `/metrics` endpoint exposes Prometheus-compatible format
- Integrates with Grafana, Prometheus for dashboards

---

## Slide 17: Deployment & Containerization

### Production-Ready Setup

**Docker Deployment:**

**Dockerfile:**
- Base: Python 3.11 slim
- Copies source, installs dependencies (`pip install -r requirements.txt`)
- Exposes port 8000
- Entrypoint: `uvicorn app:app --host 0.0.0.0 --port 8000`

**docker-compose.yml:**
- **Web Service:** FastAPI app container
- **Volumes:** Mount `vectorstore/`, `data/`, config files
- **Ports:** 8000 → localhost:8000
- **Environment:** Load from `.env` file (GROQ_API_KEY, JWT_SECRET, etc.)

**Environment Variables:**
```
GROQ_API_KEY=<your-groq-api-key>
JWT_SECRET=<your-jwt-secret>
JWT_ALGORITHM=HS256
DEFAULT_ORG_ID=public
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
ENCRYPTION_KEY=<fernet-key>
```

**Local Development:**
```bash
pip install -r requirements.txt
python ingest.py              # Build vector index
uvicorn app:app --reload --port 8000
```

---

## Slide 18: Technology Stack Summary

### Libraries & Dependencies

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI, Uvicorn |
| **LLM / Agent** | langchain, langchain-groq, ChatGroq (Llama 3.1) |
| **Embedding** | Sentence-Transformers, langchain-huggingface |
| **Vector DB** | FAISS (faiss-cpu) |
| **Reranker** | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| **Lexical Search** | rank-bm25 |
| **PDF Processing** | PyMuPDF, PyPDF2 |
| **Document Chunking** | langchain-text-splitters |
| **Database** | SQLite3 |
| **Encryption** | cryptography (Fernet) |
| **Authentication** | PyJWT |
| **Monitoring** | opentelemetry-api, opentelemetry-sdk, prometheus-client |
| **Testing** | pytest, pytest-asyncio, ragas |
| **Utilities** | httpx, numpy, datasets, python-dotenv, schedule, fpdf2 |

---

## Slide 19: Advanced Features

### Production Hallmarks

**Hallucination Prevention:**
- Enforces citation requirement for all legal claims
- Rejects answers without grounded sources
- Validates numeric claims against context
- Sanitizes LLM output to remove tool artifacts

**Multi-Tenancy:**
- Every query isolated by `org_id` from JWT token
- Data encrypted, segregated in DB
- Permissions model supports role-based access

**Caching & Performance:**
- Semantic cache reduces redundant LLM calls
- Hybrid retrieval (dense + lexical) balances precision/recall
- Adaptive retrieval depth avoids over-fetching
- CrossEncoder reranking improves relevance

**Evaluation & Feedback:**
- RAGAS metrics on every generation
- Encrypted storage of evaluations
- Hallucination tracking over time
- Confidence scoring for manual review routing

**Observability:**
- OpenTelemetry tracing for debugging
- Prometheus metrics for dashboarding
- Structured logging with request IDs
- PII redaction before logging

---

## Slide 20: Quick Start Commands

### Get Running in 5 Minutes

**1. Install Dependencies:**
```bash
cd ~/Desktop/ML-Journey/Pakistan-Legal-AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set Environment Variables:**
```bash
# Create .env file
cat > .env << EOF
GROQ_API_KEY=your-api-key-here
JWT_SECRET=your-secret-here
DEFAULT_ORG_ID=public
ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
EOF
```

**3. Ingest PDFs (Optional):**
```bash
# Put legal PDFs in data/ folder, then:
python ingest.py
```

**4. Run Application:**
```bash
uvicorn app:app --reload --port 8000
```

**5. Access:**
- API: `http://localhost:8000/docs`
- Chat: POST to `/api/v1/chat` with JWT token

---

## Slide 21: Future Enhancements

### Roadmap Ideas

**Short-term:**
- Redis backend for distributed caching
- GraphQL API layer for complex queries
- Fine-tuned reranker on Pakistani legal domain
- Additional languages (Urdu, English translation)

**Medium-term:**
- Custom legal Q&A fine-tuning on dataset
- Knowledge graph construction from legal documents
- Multi-hop reasoning for complex queries
- Audit trail & compliance reports

**Long-term:**
- Integration with official legal databases
- Real-time law update feeds
- Mobile app with offline capability
- Federation with other country legal systems

---

## Slide 22: Summary & Key Takeaways

### Pakistan Legal AI in One Picture

✅ **End-to-End Agentic RAG:** User question → agent selects tools → retrieve + generate → stream response

✅ **Grounded Answers:** All legal claims backed by official documents with citations

✅ **Production-Grade:** JWT auth, encryption, multi-tenant isolation, observability, retention policies

✅ **Hybrid Retrieval:** Dense (FAISS) + lexical (BM25) + reranking (CrossEncoder) = high precision

✅ **Evaluation Loop:** RAGAS metrics, hallucination tracking, confidence scoring, human review routing

✅ **Scalable Architecture:** Docker-ready, async/await, streaming, caching, metrics collection

✅ **Extensible:** Modular design allows easy addition of new tools, documents, or LLMs

---

## Contact & Resources

**Repository:** Pakistan-Legal-AI (Desktop/ML-Journey)

**Key Files:**
- Ingestion: `ingest.py`, `extract_and_load_pdfs.py`
- Retrieval: `retrieval_service.py`, `embedding_provider.py`
- Generation: `generation_service.py`, `pakistan_legal_assistant.py`
- API: `app.py`, `api/routes/`
- DB: `db/schema.py`, `db/retention_policy.py`
- Auth: `auth_middleware.py`
- Observability: `monitoring/`, `middleware/`

**Documentation:** `README.md`, `Pakistan_Legal_Intelligence_Full_Roadmap.md`

**Questions?** Review the code inline or refer to docstrings in modules.
