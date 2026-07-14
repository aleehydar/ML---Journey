import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from utils.logging_config import setup_logging

logger = setup_logging("pakistan-legal-ai")

from auth_middleware import AuthClaims, require_permission
from db.schema import db_schema as eval_db
from generation_service import generation_service
from retrieval_service import retrieval_service

from monitoring.tracing import init_tracing
from middleware.observability import ObservabilityMiddleware
from api.routes import chat, governance
from prometheus_client import make_asgi_app

init_tracing()

app = FastAPI(title="Pakistan Legal Assistant API", description="Production RAG Assistant with Agent UI")

# Add Middlewares
app.add_middleware(ObservabilityMiddleware)

# Read allowed origins from environment variable
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include Routers
app.include_router(chat.router)
app.include_router(governance.router)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    """Serve the main split-screen interface."""
    return templates.TemplateResponse(request=request, name="index.html")

import os
import re as re_mod
from langchain_community.document_loaders import PyMuPDFLoader

def clean_pdf_text(raw_text):
    """Post-process PyMuPDF extracted text to merge fragmented lines into flowing paragraphs."""
    lines = raw_text.split('\n')
    merged = []
    buffer = ""
    
    for line in lines:
        stripped = line.strip()
        
        # Empty line = paragraph break
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")  # preserve paragraph gap
            continue
        
        # If the line looks like a section header (numbered article, ALL CAPS title, etc.), start fresh
        if re_mod.match(r'^\d+[\.\)]\s', stripped) or (stripped.isupper() and len(stripped) > 3):
            if buffer:
                merged.append(buffer)
                buffer = ""
            buffer = stripped
            continue
        
        # Otherwise, join to the current buffer 
        if buffer:
            # If the buffer ends with a sentence-ending punctuation, start a new paragraph
            if buffer.rstrip().endswith(('.', ':', ';', '—')):
                merged.append(buffer)
                buffer = stripped
            else:
                buffer += " " + stripped
        else:
            buffer = stripped
    
    if buffer:
        merged.append(buffer)
    
    # Collapse multiple blank lines into one
    result = '\n'.join(merged)
    result = re_mod.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

# Pre-cache the raw PDF pages on server startup for the Document Explorer
global_doc_map = {}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if os.path.exists(DATA_DIR):
    logger.info("📚 Caching pristine PDF pages for Document Explorer API...")
    for pdf_file in [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]:
        try:
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, pdf_file))
            for page in loader.load():
                key = f"{pdf_file} - Page {page.metadata.get('page', 0) + 1}"
                global_doc_map[key] = clean_pdf_text(page.page_content)
        except Exception as e:
            logger.error(f"Error caching {pdf_file}: {e}")

@app.get("/documents")
async def get_documents(claims: AuthClaims = Depends(require_permission("docs:read"))):
    """Return the raw legal texts so the frontend can populate the right panel safely."""
    # If we successfully cached PDF pages directly, return them natively
    if global_doc_map and claims.org_id == "public":
        return JSONResponse(global_doc_map)
    
    return JSONResponse(retrieval_service.get_documents_for_org(claims.org_id))

@app.post("/chat")
async def chat_endpoint(
    req: QuestionRequest,
    claims: AuthClaims = Depends(require_permission("chat:write")),
):
    """Accept user question and history, returns Server-Sent Events stream."""
    logger.info(f"📊 Processing chat request for user {claims.sub} in org {claims.org_id}")
    generator = generation_service.answer_legal_question(
        req.question,
        history=req.history,
        user_id=claims.sub,
        org_id=claims.org_id,
        permissions=claims.permissions,
    )
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/api/eval/summary")
async def get_evaluation_summary(
    claims: AuthClaims = Depends(require_permission("eval:read")),
):
    """Get rolling averages for evaluation metrics including hallucination rate."""
    try:
        rolling_averages = eval_db.get_rolling_averages(claims.org_id)
        time_series_data = eval_db.get_time_series_data(claims.org_id, days=30)
        
        return JSONResponse({
            "rolling_averages": rolling_averages,
            "time_series": time_series_data,
            "org_id": claims.org_id,
        })
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to fetch evaluation summary: {str(e)}"}, 
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
