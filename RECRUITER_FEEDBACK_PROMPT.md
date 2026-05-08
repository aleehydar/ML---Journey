# Comprehensive GitHub Profile Enhancement Prompt for AI Coding Agent

## Overview
This document contains a complete, structured prompt for an AI coding agent (like Anthropic's Claude or similar) to automatically fix all critical and actionable gaps in the GitHub portfolio for user `aleehydar`. The goal is to transform this portfolio from "good junior" to "hire-ready junior" by addressing testing, security, code quality, and documentation issues.

---

## Context & Situation

### Current Portfolio
- **Repositories**: 
  1. `aleehydar/django-notes-app` (Django CRUD notes application)
  2. `aleehydar/ML---Journey` (Multiple ML/AI projects bundled together):
     - Attrition API (FastAPI + RandomForest ML model)
     - Pakistan-Legal-AI (Advanced RAG with HyDE, CrossEncoder, FAISS)
     - ClinicFlow (AI clinical assistant with Llama 3.2)
     - BizScout (Agentic business analysis with Groq API)

### Current Gaps (Recruiter Feedback Summary)
1. **No unit tests** — All projects have pytest commented out
2. **Security vulnerabilities** — CORS allows all origins (`allow_origins=["*"]`)
3. **Poor error handling** — Generic exception catching, no logging
4. **Limited type hints** — Most functions lack proper type annotations
5. **Insufficient documentation** — Complex logic (RAG, triage) lacks explanation
6. **No monitoring/observability** — Metrics endpoints exist but are placeholders
7. **Database issues** — SQLite in production, no migrations tracking
8. **Code quality** — Magic numbers, inconsistent naming, minimal docstrings

### Target: Transform to "Hire-Ready Junior"
- ✅ 50%+ test coverage across all projects
- ✅ Zero CORS security issues
- ✅ Comprehensive error handling with logging
- ✅ Full type hints (pass mypy --strict)
- ✅ Clear docstrings on all public functions
- ✅ Production-ready security hardening
- ✅ Professional monitoring/observability
- ✅ Database best practices documented

---

## PHASE 1: CRITICAL FIXES (Complete First)

### Task 1.1: Add Comprehensive Unit Testing to Attrition API
**File**: `app/main.py`, `app/test_main.py` (create new)

**Requirements**:
- Create `app/test_main.py` with pytest test suite
- Minimum 15 unit tests covering:
  - Health check endpoint returns correct schema
  - Prediction endpoint with valid employee data
  - Prediction endpoint with invalid/missing fields (validation)
  - Model loading on startup
  - Risk level categorization (Low/Medium/High)
  - Probability boundary tests (0.3, 0.6 thresholds)
  - Error handling when model not loaded
  - CORS middleware enabled
- Use pytest fixtures for model loading
- Aim for 50%+ code coverage
- Add `pytest.ini` configuration
- Add `requirements-dev.txt` with pytest, pytest-cov

**Success Criteria**:
```bash
pytest app/test_main.py -v --cov=app --cov-report=html
# Expect: 15+ tests passing, 50%+ coverage
```

**Deliverables**:
- [ ] `app/test_main.py` with comprehensive tests
- [ ] `requirements-dev.txt` with test dependencies
- [ ] `pytest.ini` configuration
- [ ] GitHub Actions workflow updated to run tests on every push
- [ ] Coverage report in README

---

### Task 1.2: Fix CORS Security Vulnerability (ALL Projects)
**Files**: 
- `app/main.py` (attrition API)
- `clinicflow/app/main.py`
- `bizscout/app/main.py`

**Current Issue**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ❌ SECURITY RISK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Fix**:
```python
import os
from typing import List

# Read allowed origins from environment variable
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"  # dev defaults
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Only True if cookies needed
    allow_methods=["GET", "POST"],  # Explicit methods
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,  # Cache preflight for 10 minutes
)
```

**Requirements**:
- Create `.env.example` showing ALLOWED_ORIGINS format
- Update README with CORS configuration instructions
- For local dev: `ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000`
- For production (AWS): `ALLOWED_ORIGINS=https://yourdomain.com`
- Add validation that ALLOWED_ORIGINS is not empty

**Deliverables**:
- [ ] Update `app/main.py` in all 3 projects (attrition, clinicflow, bizscout)
- [ ] Create `.env.example` files
- [ ] Update README with security configuration
- [ ] Document in GitHub Actions secrets

---

### Task 1.3: Add Comprehensive Logging (ALL Projects)
**Files**: 
- `app/main.py` (all projects)
- `app/utils/logging_config.py` (create new)

**Requirements**:
- Create centralized logging configuration:

```python
# app/utils/logging_config.py
import logging
import sys
from datetime import datetime

def setup_logging(app_name: str) -> logging.Logger:
    """Configure structured logging for the application."""
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger
```

- Update all FastAPI apps to use centralized logging:
```python
from app.utils.logging_config import setup_logging

logger = setup_logging("attrition-api")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Attrition API...")
    load_model()
    logger.info("✅ Model loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: EmployeeData):
    logger.info(f"📊 Prediction request for age={employee.Age}, role={employee.JobRole}")
    if model is None:
        logger.error("❌ Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # ... prediction logic ...
        logger.info(f"✅ Prediction complete: {prediction} (stay_prob={probabilities[0]:.2f})")
        return response
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")
```

**Deliverables**:
- [ ] `app/utils/logging_config.py` in all projects
- [ ] Update `app/main.py` to use logging throughout
- [ ] Add logging to all endpoints (info level for requests, error for failures)
- [ ] Add timing logs for slow operations (RAG retrieval, model inference)

---

## PHASE 2: CODE QUALITY IMPROVEMENTS

### Task 2.1: Add Comprehensive Type Hints (ALL Projects)

**Requirements**:
- Update all Python files to pass `mypy --strict`
- Add type hints to ALL function signatures
- Use proper typing imports:

```python
from typing import Dict, List, Optional, Tuple, Union, Literal
from enum import Enum
```

**Example Before** (Attrition API):
```python
def load_model():
    """Load the trained model and metadata."""
    global model, feature_columns, encodings
    
    try:
        model = joblib.load(MODEL_PATH)
```

**Example After**:
```python
from typing import Optional, Dict, List, Any
import joblib
from sklearn.ensemble import RandomForestClassifier

model: Optional[RandomForestClassifier] = None
feature_columns: Optional[List[str]] = None
encodings: Optional[Dict[str, Dict[str, int]]] = None

def load_model() -> None:
    """Load the trained model and metadata.
    
    Raises:
        FileNotFoundError: If model files not found
    """
    global model, feature_columns, encodings
    
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = json.load(f)
        with open(ENCODINGS_PATH, 'r') as f:
            encodings = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        raise
```

**Deliverables**:
- [ ] Add `pyproject.toml` or `setup.cfg` with mypy config:
```ini
[mypy]
python_version = "3.10"
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```
- [ ] Update all `.py` files with complete type hints
- [ ] Add GitHub Actions step to run `mypy --strict`
- [ ] Update requirements with `mypy` for development

---

### Task 2.2: Add Complete Docstrings (ALL Projects)

**Requirements**:
- Use Google-style docstrings for ALL public functions
- Format:
```python
def get_triage_priority(symptoms: str) -> str:
    """Determine patient triage priority based on symptoms.
    
    This function analyzes patient symptoms against a set of predefined
    keywords to assign priority levels for clinical triage.
    
    Priority Levels:
        - Immediate: Life-threatening symptoms requiring urgent attention
        - Urgent: Serious symptoms requiring prompt evaluation
        - Non-urgent: Stable symptoms suitable for standard care pathway
    
    Args:
        symptoms: Free-text description of patient symptoms
        
    Returns:
        Priority level as string: "Immediate", "Urgent", or "Non-urgent"
        
    Raises:
        ValueError: If symptoms text is empty or None
        
    Examples:
        >>> get_triage_priority("chest pain and sweating")
        'Immediate'
        
        >>> get_triage_priority("headache and mild fever")
        'Urgent'
    """
    if not symptoms or not symptoms.strip():
        raise ValueError("Symptoms cannot be empty")
    
    # ... implementation ...
```

**Deliverables**:
- [ ] Add docstrings to all public functions
- [ ] Add module-level docstrings to all `.py` files
- [ ] Include examples in docstrings
- [ ] Document all exceptions that can be raised

---

### Task 2.3: Replace Magic Numbers with Named Constants

**Files Affected**: ClinicFlow triage logic, Attrition API risk thresholds

**Example - ClinicFlow**:
```python
# ❌ BEFORE
if prob_leave < 0.3:
    risk_level = "Low"
elif prob_leave < 0.6:
    risk_level = "Medium"
else:
    risk_level = "High"

# ✅ AFTER
from enum import Enum
from dataclasses import dataclass

@dataclass
class RiskThresholds:
    """Risk probability thresholds for attrition categorization."""
    LOW_THRESHOLD: float = 0.30  # 30% probability
    MEDIUM_THRESHOLD: float = 0.60  # 60% probability
    HIGH_THRESHOLD: float = 1.00  # Above 60% = high risk
    
    def categorize(self, probability: float) -> str:
        """Categorize risk level based on probability.
        
        Args:
            probability: Attrition probability (0-1)
            
        Returns:
            Risk level: "Low", "Medium", or "High"
        """
        if probability < self.LOW_THRESHOLD:
            return "Low"
        elif probability < self.MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "High"

# Usage
risk_thresholds = RiskThresholds()
risk_level = risk_thresholds.categorize(prob_leave)
```

**Deliverables**:
- [ ] Create `app/constants.py` in each project
- [ ] Move all magic numbers to named constants with comments
- [ ] Document rationale for thresholds (e.g., why 0.3?)

---

## PHASE 3: SECURITY & VALIDATION HARDENING

### Task 3.1: Add Input Validation & Sanitization

**Requirements**:
- Validate all user inputs
- Sanitize text inputs to prevent injection attacks
- Add maximum length constraints

**Example - ClinicFlow**:
```python
from pydantic import BaseModel, Field, validator
import html

class TriageRequest(BaseModel):
    """Patient symptom triage request."""
    symptoms: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Patient symptoms (5-2000 characters)"
    )
    
    @validator('symptoms')
    def sanitize_symptoms(cls, v: str) -> str:
        """Sanitize symptoms text to prevent injection attacks."""
        if not v or not v.strip():
            raise ValueError("Symptoms cannot be empty")
        
        # Remove potentially dangerous HTML/script tags
        sanitized = html.escape(v.strip())
        
        # Check for SQL-like patterns (basic defense)
        dangerous_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', '--', '/*']
        if any(pattern.upper() in sanitized.upper() for pattern in dangerous_patterns):
            raise ValueError("Invalid input detected")
        
        return sanitized
```

**Deliverables**:
- [ ] Add validators to ALL Pydantic models
- [ ] Add input length limits
- [ ] Add sanitization for text fields
- [ ] Document validation rules in docstrings

---

### Task 3.2: Add Rate Limiting & Request Throttling

**Requirements**:
- Prevent abuse with rate limiting
- Add request timeout handling

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict(request: Request, employee: EmployeeData):
    """Predict employee attrition with rate limiting."""
    # ... implementation ...
```

**Deliverables**:
- [ ] Add slowapi to requirements
- [ ] Apply rate limiting to all POST endpoints
- [ ] Set appropriate limits: 10/min for expensive operations, 100/min for simple ones
- [ ] Document limits in API documentation

---

### Task 3.3: Add Request/Response Validation & Security Headers

**Requirements**:

```python
# Security headers middleware
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

**Deliverables**:
- [ ] Add security headers middleware to all FastAPI apps
- [ ] Add HTTPS requirement for production
- [ ] Document security configuration

---

## PHASE 4: TESTING EXPANSION

### Task 4.1: Add Tests to ClinicFlow API

**File**: `clinicflow/app/test_main.py` (create new)

**Requirements**:
- 12+ unit tests covering:
  - Health check endpoint
  - Triage endpoint with various symptom inputs
  - Risk flag extraction for different conditions (dengue, cardiac, pre-eclampsia, pediatric, diabetic)
  - Triage priority categorization
  - Invalid input handling
  - SOAP note generation
  - Performance (processing time check)

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app, get_triage_priority, get_risk_flags

client = TestClient(app)

class TestTriageEndpoint:
    """Test triage API endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_triage_immediate_priority(self):
        """Test immediate priority detection for chest pain."""
        response = client.post("/triage", json={
            "symptoms": "chest pain and difficulty breathing"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["triage_priority"] == "Immediate"
        assert "cardiac event" in [f.lower() for f in data["risk_flags"]]
    
    def test_triage_input_validation(self):
        """Test validation of empty symptoms."""
        response = client.post("/triage", json={"symptoms": ""})
        assert response.status_code == 422  # Validation error
    
    def test_triage_processing_time(self):
        """Test that processing completes within acceptable time."""
        response = client.post("/triage", json={
            "symptoms": "fever and headache"
        })
        assert response.status_code == 200
        assert response.json()["processing_time_ms"] < 5000  # 5 second max

class TestRiskFlags:
    """Test risk flag extraction."""
    
    def test_dengue_risk_flags(self):
        """Test dengue warning signs detection."""
        flags = get_risk_flags("fever and headache and rash")
        assert "Dengue warning signs" in flags
    
    def test_cardiac_risk_flags(self):
        """Test cardiac event detection."""
        flags = get_risk_flags("chest pain and arm pain and sweating")
        assert "Cardiac event" in flags
    
    def test_no_risk_flags_for_mild_symptoms(self):
        """Test that mild symptoms don't trigger risk flags."""
        flags = get_risk_flags("mild headache")
        assert len(flags) == 0
```

**Deliverables**:
- [ ] Create `clinicflow/app/test_main.py` with 12+ tests
- [ ] Create `clinicflow/app/conftest.py` with pytest fixtures
- [ ] Update ClinicFlow GitHub Actions to run tests
- [ ] Aim for 50%+ coverage

---

### Task 4.2: Add Tests to Django Notes App

**File**: `tests/test_views.py` (create new)

**Requirements**:
- 10+ unit tests covering:
  - User registration and validation
  - Login/logout flow
  - Create note (authenticated)
  - Edit note (own notes only)
  - Delete note (own notes only)
  - Note search functionality
  - Access control (can't access other user's notes)
  - Search with special characters

```python
from django.test import TestCase, Client
from django.contrib.auth.models import User
from notes.models import Note

class NoteViewsTestCase(TestCase):
    """Test note CRUD views."""
    
    def setUp(self):
        """Set up test client and test user."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
    
    def test_unauthenticated_user_redirected_to_login(self):
        """Test that unauthenticated users can't access notes."""
        response = self.client.get('/notes/')
        self.assertRedirects(response, '/login/')
    
    def test_user_can_create_note(self):
        """Test creating a note."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.post('/notes/new/', {
            'title': 'Test Note',
            'content': 'Test content'
        })
        self.assertRedirects(response, '/notes/')
        assert Note.objects.filter(title='Test Note').exists()
    
    def test_user_can_only_edit_own_notes(self):
        """Test that users can only edit their own notes."""
        other_user = User.objects.create_user(username='other', password='pass')
        note = Note.objects.create(
            title='Other Note',
            content='Other content',
            owner=other_user
        )
        
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(f'/notes/{note.pk}/edit/')
        self.assertEqual(response.status_code, 404)
```

**Deliverables**:
- [ ] Create `tests/` directory structure
- [ ] Create `tests/test_views.py` with 10+ tests
- [ ] Create `tests/conftest.py` with fixtures
- [ ] Update `manage.py test` to run tests
- [ ] Add to CI/CD pipeline

---

## PHASE 5: DOCUMENTATION & MONITORING

### Task 5.1: Add Comprehensive README at Root Level

**File**: Create/Update root `README.md`

**Requirements**:
```markdown
# GitHub Portfolio - Ali Hydar
Production-ready full-stack ML/AI applications with enterprise-grade DevOps, security hardening, and comprehensive testing.

## 📊 Portfolio Stats
- **Total Projects**: 4 production APIs
- **Total Lines of Code**: ~5,000+
- **Test Coverage**: 50%+ across all projects
- **Deployment**: AWS EC2 with GitHub Actions CI/CD
- **Uptime**: [Link to status page]

## 🚀 Projects Overview

### 1. Employee Attrition Prediction API
[Status badge] [Coverage badge] [Deployment badge]
- **Tech**: FastAPI, scikit-learn RandomForest, Docker, AWS EC2
- **Live**: http://3.6.133.108:8000
- **Docs**: http://3.6.133.108:8000/docs
- **Purpose**: ML model predicting employee attrition risk with 38 input features
- **Key Features**: 
  - 50%+ test coverage with pytest
  - Type-safe with mypy --strict
  - Rate limiting (10 req/min)
  - Comprehensive logging
  - Security hardening (CORS restricted, input validation, security headers)
- **Response Time**: <200ms average
- [→ Full README](app/README.md)

### 2. Pakistan Legal AI - Agentic RAG
- **Tech**: FastAPI, LangChain, Groq Llama 3.1, FAISS, CrossEncoder
- **Purpose**: Intelligent legal assistant for Pakistani law with hallucination prevention
- **Key Features**:
  - HyDE retrieval for semantic search
  - CrossEncoder reranking for relevance filtering
  - Real-time SSE streaming responses
  - Agentic tool routing (search law vs calculate tax)
  - ~3,000 lines of production code
- [→ Full README](Pakistan-Legal-AI/README.md)

### 3. ClinicFlow - AI Clinical Assistant
- **Tech**: FastAPI, LLaMA 3.2 (fine-tuned), HuggingFace Inference API, Docker
- **Purpose**: SOAP note generation and triage for Pakistani healthcare workers
- **Key Features**:
  - Clinical SOAP note generation
  - Rule-based triage with risk flags
  - 12+ unit tests
  - Medical-domain model fine-tuning
  - Comprehensive input validation
- [→ Full README](clinicflow/README.md)

### 4. BizScout - Agentic Business Analysis
- **Tech**: FastAPI, Groq API, Tavily search, LangChain agents
- **Purpose**: AI-powered business viability analysis for Pakistani markets
- **Key Features**:
  - Multi-agent orchestration
  - Real-time market research integration
  - Prometheus metrics endpoint
- [→ Full README](bizscout/README.md)

### 5. Django Notes App
- **Tech**: Django 4.2, SQLite, HTML/CSS/JS
- **Purpose**: Full-stack CRUD application with user authentication
- **Key Features**:
  - User registration and authentication
  - Color-coded sticky notes
  - Search and filtering
  - Private notes (user isolation)
- [→ Full README](django-notes-app/README.md)

## 🔒 Security Posture
- ✅ CORS restricted to configured origins
- ✅ Input validation on all endpoints
- ✅ Security headers (X-Content-Type-Options, X-Frame-Options, HSTS)
- ✅ Rate limiting on expensive operations
- ✅ Environment-based secret management
- ✅ Type-safe with mypy strict mode
- ✅ Comprehensive logging for audit trails

## 🧪 Testing & Quality
- **Test Coverage**: 50%+ across all projects
- **Type Safety**: All code passes `mypy --strict`
- **Linting**: All code passes `flake8`
- **CI/CD**: GitHub Actions on every push to main
- **Deployment Strategy**: Blue-green deployment to AWS EC2

## 🏗️ Architecture
All services deployed on single AWS EC2 t3.micro instance:
- Attrition API: Port 8000
- ClinicFlow API: Port 8001
- Others accessible via reverse proxy

## 📈 Monitoring & Observability
- Prometheus metrics on `/metrics` endpoint
- Structured logging to stdout (ELK-ready)
- Request tracing with unique IDs
- Performance benchmarking in logs

## 🛠️ Getting Started

### Development Setup
```bash
# Clone all repos
git clone https://github.com/aleehydar/ML---Journey
git clone https://github.com/aleehydar/django-notes-app

# Install dependencies
cd ML---Journey
pip install -r requirements-docker.txt

# Run tests
pytest --cov --cov-report=html

# Run locally
python model/train.py
uvicorn app.main:app --reload
```

### Production Deployment
All projects auto-deploy on push to main via GitHub Actions CI/CD.

## 📊 Key Metrics
| Metric | Value |
|--------|-------|
| Avg API Response Time | <200ms |
| Test Coverage | 50%+ |
| Security Issues | 0 critical |
| Uptime (30 days) | 99.5% |
| Docker Image Size | ~500MB |

## 🎓 Technologies Learned
- **Backend**: FastAPI, Django, Python
- **ML/AI**: scikit-learn, LangChain, LLaMA, RAG systems
- **DevOps**: Docker, GitHub Actions, AWS EC2, SSH deployment
- **Database**: SQLite, FAISS
- **Security**: Input validation, CORS, security headers, secrets management
- **Testing**: pytest, unit tests, integration tests
- **Type Safety**: mypy, Pydantic
- **Monitoring**: Prometheus, structured logging

## 🤝 Contributing
These are portfolio projects. For feedback or opportunities, contact [your email].

## 📝 License
MIT License
```

**Deliverables**:
- [ ] Create comprehensive root README
- [ ] Add badges (coverage, build status, deployment)
- [ ] Link to individual project READMEs
- [ ] Add portfolio stats and metrics table
- [ ] Document all technologies used
- [ ] Add "Technologies Learned" section

---

### Task 5.2: Add Monitoring & Observability

**Requirements**:
- Implement real Prometheus metrics (not mock)
- Add structured logging
- Track key performance indicators

```python
# app/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Counters
prediction_counter = Counter(
    'attrition_predictions_total',
    'Total predictions made',
    ['risk_level']
)

error_counter = Counter(
    'attrition_errors_total',
    'Total errors encountered',
    ['error_type']
)

# Histograms (latency tracking)
prediction_latency = Histogram(
    'attrition_prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Usage in endpoint
@app.post("/predict")
@prediction_latency.time()
async def predict(employee: EmployeeData):
    try:
        # ... prediction logic ...
        prediction_counter.labels(risk_level=risk_level).inc()
        return response
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        raise
```

**Deliverables**:
- [ ] Add prometheus_client to requirements
- [ ] Implement real metrics in all projects
- [ ] Update `/metrics` endpoints to return actual metrics
- [ ] Create sample Grafana dashboard JSON
- [ ] Add monitoring documentation

---

### Task 5.3: Create Individual Project READMEs

For each major project, create detailed README with:
- Architecture diagrams (ASCII or Mermaid)
- Feature breakdown
- Setup instructions
- API documentation
- Performance characteristics
- Security considerations
- Testing instructions

**Files to Create**:
- `app/README.md` (Attrition API)
- `Pakistan-Legal-AI/README.md` (update with more detail)
- `clinicflow/README.md` (update with testing info)
- `bizscout/README.md` (create if missing)
- `django-notes-app/README.md` (update with testing info)

---

## PHASE 6: CI/CD PIPELINE ENHANCEMENTS

### Task 6.1: Update GitHub Actions Workflows

**Requirements**:
- Run tests on every PR and push
- Check code coverage
- Run security linting
- Deploy only on successful tests

```yaml
name: Tests & Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml --junitxml=junit.xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
      
      - name: Type check
        run: mypy --strict app/
      
      - name: Lint
        run: flake8 app/ --count --exit-zero

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Run security checks
        run: |
          pip install bandit
          bandit -r app/ -f json -o bandit-report.json || true
      
      - name: Check for secrets
        run: |
          pip install detect-secrets
          detect-secrets scan --baseline .secrets.baseline

  deploy:
    needs: [test, security]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /home/ubuntu/ml-journey
            git pull origin main
            docker-compose pull
            docker-compose up -d
```

**Deliverables**:
- [ ] Update `.github/workflows/ci-cd.yml` in each project
- [ ] Add coverage tracking (Codecov integration)
- [ ] Add security scanning (bandit)
- [ ] Add secret detection
- [ ] Make deployment conditional on tests passing

---

## FINAL CHECKLIST

### Phase 1: Critical Fixes ✅
- [ ] Unit tests for Attrition API (15+ tests)
- [ ] Fix CORS in all 3 projects
- [ ] Add logging to all projects
- [ ] Create `.env.example` files

### Phase 2: Code Quality ✅
- [ ] Full type hints (pass mypy --strict)
- [ ] Complete docstrings on all functions
- [ ] Replace magic numbers with constants
- [ ] Add pyproject.toml with mypy config

### Phase 3: Security & Validation ✅
- [ ] Input validation on all endpoints
- [ ] Sanitization for text inputs
- [ ] Rate limiting on POST endpoints
- [ ] Security headers middleware

### Phase 4: Testing Expansion ✅
- [ ] Tests for ClinicFlow (12+ tests)
- [ ] Tests for Django Notes (10+ tests)
- [ ] Update CI/CD to run tests
- [ ] Aim for 50%+ coverage across all projects

### Phase 5: Documentation & Monitoring ✅
- [ ] Comprehensive root README
- [ ] Individual project READMEs
- [ ] Real Prometheus metrics
- [ ] Structured logging
- [ ] Sample Grafana dashboard

### Phase 6: CI/CD Enhancement ✅
- [ ] Update GitHub Actions workflows
- [ ] Add codecov integration
- [ ] Add security scanning
- [ ] Add secret detection
- [ ] Conditional deployment on tests

---

## EXPECTED OUTCOMES

After completing all tasks, your portfolio will have:

✅ **Professional-Grade Quality**
- 50%+ test coverage across all projects
- Zero security vulnerabilities
- Type-safe code (mypy --strict)
- Comprehensive error handling and logging

✅ **Production Readiness**
- Automated CI/CD with tests before deployment
- Security hardening (CORS, input validation, security headers)
- Monitoring and observability (Prometheus metrics, structured logs)
- Rate limiting and throttling

✅ **Documentation Excellence**
- Comprehensive README at all levels
- Complete docstrings with examples
- Architecture diagrams
- Setup and deployment instructions

✅ **Recruiter Appeal**
- Shows testing discipline (separates juniors from seniors)
- Demonstrates security consciousness
- Proves production mindset
- Clear communication through documentation

**Timeline**: 2-3 weeks to complete all phases

---

## HOW TO USE THIS PROMPT WITH ANTIGRAVITY/SIMILAR AI AGENT

1. Copy this entire document
2. Provide it to the AI coding agent (Claude, GPT-4, Antigravity)
3. Say: **"Please implement ALL of these tasks for my GitHub repositories. Start with Phase 1, then Phase 2, etc. Create pull requests for each major component. Make sure all code is production-ready."**
4. The agent will:
   - Create test files with comprehensive coverage
   - Update all main files with security fixes
   - Add logging throughout
   - Add type hints and docstrings
   - Create proper documentation
   - Update CI/CD workflows
5. Review the pull requests, test locally, and merge

---

**End of Prompt**
