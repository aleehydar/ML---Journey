"""
FastAPI Application for Employee Attrition Prediction
Provides endpoints for health checks and predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import os
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition risk using a trained RandomForest model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
ENCODINGS_PATH = os.path.join(MODEL_DIR, "encodings.json")

# Global variables for model and metadata
model = None
feature_columns = None
encodings = None


def load_model():
    """Load the trained model and metadata."""
    global model, feature_columns, encodings
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = json.load(f)
        print(f"✅ Feature columns loaded: {len(feature_columns)} features")
        
        with open(ENCODINGS_PATH, 'r') as f:
            encodings = json.load(f)
        print(f"✅ Encodings loaded for {len(encodings)} categorical variables")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Please run 'python model/train.py' first to train the model.")
        raise


# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()


# Request/Response schemas
class EmployeeData(BaseModel):
    """Employee data for attrition prediction."""
    Age: int = Field(..., ge=18, le=65, description="Employee age in years")
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    DailyRate: int = Field(..., ge=100, le=1500, description="Daily rate")
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    DistanceFromHome: int = Field(..., ge=1, le=30, description="Distance from home in km")
    Education: int = Field(..., ge=1, le=5, description="Education level (1=Below College, 5=Doctor)")
    EducationField: Literal["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1-4)")
    Gender: Literal["Male", "Female"]
    HourlyRate: int = Field(..., ge=30, le=100, description="Hourly rate")
    JobInvolvement: int = Field(..., ge=1, le=4, description="Job involvement (1-4)")
    JobLevel: int = Field(..., ge=1, le=5, description="Job level (1-5)")
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1-4)")
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: int = Field(..., ge=1000, le=20000, description="Monthly income in USD")
    MonthlyRate: int = Field(..., ge=1000, le=25000, description="Monthly rate")
    NumCompaniesWorked: int = Field(..., ge=0, le=9, description="Number of companies worked")
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(..., ge=0, le=25, description="Percent salary hike")
    PerformanceRating: int = Field(..., ge=1, le=4, description="Performance rating (1-4)")
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="Relationship satisfaction (1-4)")
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Stock option level (0-3)")
    TotalWorkingYears: int = Field(..., ge=0, le=40, description="Total working years")
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, description="Training times last year")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work life balance (1-4)")
    YearsAtCompany: int = Field(..., ge=0, le=40, description="Years at company")
    YearsInCurrentRole: int = Field(..., ge=0, le=18, description="Years in current role")
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15, description="Years since last promotion")
    YearsWithCurrManager: int = Field(..., ge=0, le=17, description="Years with current manager")


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: int = Field(..., description="0 = Stay, 1 = Leave")
    probability_stay: float = Field(..., description="Probability of staying (0-1)")
    probability_leave: float = Field(..., description="Probability of leaving (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API info and health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: EmployeeData):
    """
    Predict employee attrition risk.
    
    Returns:
        - prediction: 0 (stay) or 1 (leave)
        - probability_stay: Probability of staying
        - probability_leave: Probability of leaving
        - risk_level: Categorized risk level
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = []
        
        for col in feature_columns:
            value = getattr(employee, col)
            
            # Encode categorical variables
            if col in encodings:
                value = encodings[col].get(value, 0)
            
            input_data.append(value)
        
        # Convert to numpy array
        input_array = np.array([input_data])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]
        
        # Determine risk level
        prob_leave = probabilities[1]
        if prob_leave < 0.3:
            risk_level = "Low"
        elif prob_leave < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability_stay=float(probabilities[0]),
            probability_leave=float(probabilities[1]),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get model information and feature list."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "categorical_variables": list(encodings.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
