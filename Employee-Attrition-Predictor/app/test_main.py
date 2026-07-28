import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Mocking before importing the app to avoid loading the real model if not available
with patch('joblib.load') as mock_joblib, \
     patch('builtins.open', create=True) as mock_open, \
     patch('json.load') as mock_json_load:
    
    from app.main import app, load_model
    import app.main as main_module

client = TestClient(app)

# Sample valid employee data
VALID_EMPLOYEE = {
    "Age": 30,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
}

@pytest.fixture
def mock_model():
    """Mock the machine learning model."""
    mock = MagicMock()
    mock.n_estimators = 100
    mock.max_depth = 10
    # Default to "stay" (0) prediction
    mock.predict.return_value = np.array([0])
    mock.predict_proba.return_value = np.array([[0.8, 0.2]])
    return mock

@pytest.fixture
def loaded_app_state(mock_model):
    """Fixture to mock the model being loaded in the application state."""
    main_module.model = mock_model
    main_module.feature_columns = ["Age", "DailyRate", "Department"]
    main_module.encodings = {"Department": {"Research & Development": 1}}
    yield main_module
    main_module.model = None
    main_module.feature_columns = None
    main_module.encodings = None

class TestHealthCheck:
    def test_root_endpoint(self, loaded_app_state):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True
        
    def test_health_check_endpoint(self, loaded_app_state):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

    def test_health_check_no_model(self):
        main_module.model = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"
        assert response.json()["model_loaded"] is False

class TestPredictionValidation:
    def test_prediction_valid_data(self, loaded_app_state):
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability_stay" in data
        assert "probability_leave" in data
        assert "risk_level" in data

    def test_prediction_missing_field(self, loaded_app_state):
        invalid_employee = VALID_EMPLOYEE.copy()
        del invalid_employee["Age"]
        response = client.post("/predict", json=invalid_employee)
        assert response.status_code == 422  # Validation Error

    def test_prediction_invalid_bounds(self, loaded_app_state):
        invalid_employee = VALID_EMPLOYEE.copy()
        invalid_employee["Age"] = 15  # Below min 18
        response = client.post("/predict", json=invalid_employee)
        assert response.status_code == 422

    def test_prediction_invalid_category(self, loaded_app_state):
        invalid_employee = VALID_EMPLOYEE.copy()
        invalid_employee["Department"] = "Invalid Dept"
        response = client.post("/predict", json=invalid_employee)
        assert response.status_code == 422

class TestModelState:
    def test_predict_model_not_loaded(self):
        main_module.model = None
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    @patch('app.main.joblib.load')
    def test_load_model_failure(self, mock_load):
        mock_load.side_effect = FileNotFoundError("Model file missing")
        with pytest.raises(FileNotFoundError):
            load_model()

    def test_model_info_loaded(self, loaded_app_state):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert response.json()["feature_count"] == 3

    def test_model_info_not_loaded(self):
        main_module.model = None
        response = client.get("/model-info")
        assert response.status_code == 503

class TestRiskLevels:
    def test_low_risk(self, loaded_app_state):
        loaded_app_state.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.json()["risk_level"] == "Low"

    def test_medium_risk_boundary_1(self, loaded_app_state):
        loaded_app_state.model.predict_proba.return_value = np.array([[0.65, 0.35]])
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.json()["risk_level"] == "Medium"
        
    def test_medium_risk_boundary_2(self, loaded_app_state):
        loaded_app_state.model.predict_proba.return_value = np.array([[0.41, 0.59]])
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.json()["risk_level"] == "Medium"

    def test_high_risk(self, loaded_app_state):
        loaded_app_state.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        response = client.post("/predict", json=VALID_EMPLOYEE)
        assert response.json()["risk_level"] == "High"

class TestSecurity:
    def test_cors_headers(self, loaded_app_state):
        response = client.options(
            "/predict", 
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
        )
        assert response.status_code == 200
        # CORSMiddleware responds with headers if origin is allowed
        assert "access-control-allow-origin" in response.headers
