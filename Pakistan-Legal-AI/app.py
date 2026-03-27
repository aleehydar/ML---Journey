from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from pakistan_legal_assistant import answer_legal_question, legal_texts, vectorstore

app = FastAPI(title="Pakistan Legal Assistant API", description="Production RAG Assistant with Agent UI")

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
    print("📚 Caching pristine PDF pages for Document Explorer API...")
    for pdf_file in [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]:
        try:
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, pdf_file))
            for page in loader.load():
                key = f"{pdf_file} - Page {page.metadata.get('page', 0) + 1}"
                global_doc_map[key] = clean_pdf_text(page.page_content)
        except Exception as e:
            print(f"Error caching {pdf_file}: {e}")

@app.get("/documents")
async def get_documents():
    """Return the raw legal texts so the frontend can populate the right panel safely."""
    # If we successfully cached PDF pages directly, return them natively
    if global_doc_map:
        return JSONResponse(global_doc_map)
    
    # Fallback to hardcoded sample texts if FAISS and PDFs are missing
    return JSONResponse({item["source"]: item["text"] for item in legal_texts})

@app.post("/chat")
async def chat_endpoint(req: QuestionRequest):
    """Accept user question and history, returns Server-Sent Events stream."""
    generator = answer_legal_question(req.question, history=req.history)
    return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
