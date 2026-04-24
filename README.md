# MLOps Attrition Prediction API

[![CI/CD Pipeline](https://github.com/aleehydar/ML-Journey/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/aleehydar/ML-Journey/actions/workflows/ci-cd.yml)

A production-ready Machine Learning API for predicting employee attrition risk, built with **FastAPI**, **Docker**, **GitHub Actions CI/CD**, and deployed on **AWS EC2**.

## Live API

The API is deployed and running at:

**http://3.6.133.108:8000/**

### Interactive Documentation
- **Swagger UI:** http://3.6.133.108:8000/docs
- **ReDoc:** http://3.6.133.108:8000/redoc

### Quick Test

**Health Check:**
```bash
curl http://3.6.133.108:8000/health
```

**Prediction Example:**
```bash
curl -X POST http://3.6.133.108:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 30, "BusinessTravel": "Travel_Rarely", "DailyRate": 500, "Department": "Sales", "DistanceFromHome": 10, "Education": 3, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Sales Executive", "JobSatisfaction": 3, "MaritalStatus": "Single", "MonthlyIncome": 5000, "MonthlyRate": 10000, "NumCompaniesWorked": 1, "OverTime": "No", "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 0, "TotalWorkingYears": 5, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 3, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2}'
```

## Tech Stack

- **Framework:** FastAPI
- **ML Library:** scikit-learn (RandomForest)
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud:** AWS EC2 (t3.micro, free tier)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │────▶│  GitHub     │────▶│   Docker    │
│   (Code)    │     │  Actions    │     │    Hub      │
└─────────────┘     └─────────────┘     └──────┬──────┘
       │                                         │
       │                                         ▼
       │                                 ┌─────────────┐
       │                                 │   AWS EC2   │
       │                                 │ (Production)│
       │                                 └─────────────┘
       │
       └───────────────────────────────────────────────▶
                    Auto-deploy on push to main
```

## Local Development

### Setup
```bash
# Clone the repo
git clone https://github.com/aleehydar/ML-Journey.git
cd ML-Journey

# Install dependencies
pip install -r requirements-docker.txt

# Train the model
python model/train.py

# Run the API locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker
```bash
# Build image
docker build -t attrition-api .

# Run container
docker run -d -p 8000:8000 --restart always --name attrition-api attrition-api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + API info |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Predict attrition risk |
| `/docs` | GET | Swagger UI (interactive) |

## CI/CD Pipeline

Every push to `main` triggers:
1. **Test** — Run pytest suite
2. **Build** — Build Docker image
3. **Push** — Push to Docker Hub
4. **Deploy** — Auto-deploy to AWS EC2 via SSH

## Author

- **Ali Hydar** — [GitHub](https://github.com/aleehydar)
