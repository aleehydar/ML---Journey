# Employee Attrition Prediction API

Production-ready FastAPI application that predicts employee attrition risk using a trained RandomForest machine learning model.

## Features
- **FastAPI Backend**: High-performance asynchronous API
- **scikit-learn**: RandomForestClassifier for robust predictions
- **Security Hardened**: CORS restricted, Security headers, Rate Limiting (10 req/min)
- **Observability**: Centralized logging and Prometheus metrics
- **Quality Assured**: 50%+ pytest coverage and mypy strict typing

## Setup
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running the API
```bash
uvicorn app.main:app --reload
```

## Testing
```bash
pytest app/test_main.py -v --cov=app
```

## Architecture
The API loads a pre-trained model on startup. Incoming prediction requests are validated by Pydantic models. Predictions are converted into probability arrays to determine risk categorizations ("Low", "Medium", "High").
