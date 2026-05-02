import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from app.llm import load_model, generate_soap

app = FastAPI(title="ClinicFlow AI Clinical Assistant")

# Set up templates
# Since the app might be run from the clinicflow directory, we ensure the path is absolute relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model at startup
@app.on_event("startup")
async def startup_event():
    print("Initializing ClinicFlow LLM Model on Startup...")
    load_model()

class TriageRequest(BaseModel):
    symptoms: str

def get_triage_priority(symptoms: str) -> str:
    symptoms_lower = symptoms.lower()
    
    immediate_keywords = ["chest pain", "unconscious", "seizure", "not breathing", "severe bleeding", "bp>160", "bp > 160", "bp >160"]
    urgent_keywords = ["high fever >39", "high fever > 39", "breathing difficulty", "pregnancy + high bp", "altered consciousness"]
    
    for kw in immediate_keywords:
        if kw in symptoms_lower:
            return "Immediate"
            
    for kw in urgent_keywords:
        if kw in symptoms_lower:
            return "Urgent"
            
    return "Non-urgent"

def get_risk_flags(symptoms: str) -> list[str]:
    symptoms_lower = symptoms.lower()
    flags = []
    
    # Dengue warning signs: fever + headache + rash or retro-orbital
    if "fever" in symptoms_lower and "headache" in symptoms_lower and ("rash" in symptoms_lower or "retro-orbital" in symptoms_lower):
        flags.append("Dengue warning signs")
        
    # Cardiac event: chest pain + arm/jaw radiation or sweating
    if "chest pain" in symptoms_lower and ("arm" in symptoms_lower or "jaw" in symptoms_lower or "sweating" in symptoms_lower):
        flags.append("Cardiac event")
        
    # Pre-eclampsia: pregnant + high BP or headache or vision
    if "pregnan" in symptoms_lower and ("high bp" in symptoms_lower or "headache" in symptoms_lower or "vision" in symptoms_lower):
        flags.append("Pre-eclampsia")
        
    # Pediatric emergency: child/infant + fever>39 or fast breathing
    if ("child" in symptoms_lower or "infant" in symptoms_lower) and ("fever>39" in symptoms_lower or "fever > 39" in symptoms_lower or "fever >39" in symptoms_lower or "fast breathing" in symptoms_lower):
        flags.append("Pediatric emergency")
        
    # Diabetic emergency: diabetic + confusion or wound or high sugar
    if "diabetic" in symptoms_lower and ("confusion" in symptoms_lower or "wound" in symptoms_lower or "high sugar" in symptoms_lower):
        flags.append("Diabetic emergency")
        
    return flags

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/triage")
async def triage(req: TriageRequest):
    start_time = time.time()
    
    priority = get_triage_priority(req.symptoms)
    flags = get_risk_flags(req.symptoms)
    
    soap_note = generate_soap(req.symptoms)
    
    process_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "soap_note": soap_note,
        "triage_priority": priority,
        "risk_flags": flags,
        "processing_time_ms": process_time_ms
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    # A simple mock endpoint for Prometheus.
    # In a real scenario, use prometheus_client library
    return {
        "clinicflow_requests_total": 1,
        "clinicflow_model_loaded": 1
    }
