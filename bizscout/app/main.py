from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, Any
import time
from app.agents.orchestrator import run_analysis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BizScout Pakistan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_CITIES = [
    "Karachi", "Lahore", "Islamabad", "Peshawar", 
    "Quetta", "Multan", "Faisalabad", "Rawalpindi"
]

class AnalyzeRequest(BaseModel):
    business_idea: str
    city: str
    budget_pkr: int

    @validator("city")
    def validate_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"City must be one of: {', '.join(VALID_CITIES)}")
        return v

prediction_count = 0
total_latency = 0.0

@app.post("/analyze")
async def analyze_business(req: AnalyzeRequest) -> Dict[str, Any]:
    global prediction_count, total_latency
    start_time = time.time()
    
    try:
        report = run_analysis(req.business_idea, req.city, req.budget_pkr)
        
        latency = time.time() - start_time
        prediction_count += 1
        total_latency += latency
        
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    avg_latency = total_latency / prediction_count if prediction_count > 0 else 0.0
    prometheus_metrics = f"""# HELP bizscout_predictions_total Total number of predictions
# TYPE bizscout_predictions_total counter
bizscout_predictions_total {prediction_count}
# HELP bizscout_avg_latency_seconds Average prediction latency
# TYPE bizscout_avg_latency_seconds gauge
bizscout_avg_latency_seconds {avg_latency:.4f}
"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(prometheus_metrics)
