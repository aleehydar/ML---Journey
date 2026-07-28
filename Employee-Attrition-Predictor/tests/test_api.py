"""
API Tests for Employee Attrition Prediction
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, load_model
import os
import json

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model is loaded before running tests."""
    # Change to project root to find model files
    original_dir = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Load model if not already loaded
    try:
        load_model()
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    yield
    
    # Restore original directory
    os.chdir(original_dir)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data


class TestPredictionEndpoint:
    """Tests for the /predict endpoint."""
    
    @pytest.fixture
    def sample_employee_low_risk(self):
        """Sample employee data with low attrition risk."""
        return {
            "Age": 45,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 800,
            "Department": "Research & Development",
            "DistanceFromHome": 5,
            "Education": 4,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 4,
            "Gender": "Male",
            "HourlyRate": 75,
            "JobInvolvement": 4,
            "JobLevel": 4,
            "JobRole": "Research Director",
            "JobSatisfaction": 4,
            "MaritalStatus": "Married",
            "MonthlyIncome": 15000,
            "MonthlyRate": 20000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 20,
            "PerformanceRating": 4,
            "RelationshipSatisfaction": 4,
            "StockOptionLevel": 3,
            "TotalWorkingYears": 20,
            "TrainingTimesLastYear": 5,
            "WorkLifeBalance": 4,
            "YearsAtCompany": 15,
            "YearsInCurrentRole": 10,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 10
        }
    
    @pytest.fixture
    def sample_employee_high_risk(self):
        """Sample employee data with high attrition risk."""
        return {
            "Age": 25,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 300,
            "Department": "Sales",
            "DistanceFromHome": 25,
            "Education": 2,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 35,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2000,
            "MonthlyRate": 3000,
            "NumCompaniesWorked": 5,
            "OverTime": "Yes",
            "PercentSalaryHike": 5,
            "PerformanceRating": 2,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 3,
            "TrainingTimesLastYear": 0,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 1
        }
    
    def test_predict_low_risk(self, sample_employee_low_risk):
        """Test prediction for low-risk employee."""
        response = client.post("/predict", json=sample_employee_low_risk)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability_stay" in data
        assert "probability_leave" in data
        assert "risk_level" in data
        
        # Verify data types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability_stay"], float)
        assert isinstance(data["probability_leave"], float)
        assert isinstance(data["risk_level"], str)
        
        # Probabilities should sum to approximately 1
        total_prob = data["probability_stay"] + data["probability_leave"]
        assert abs(total_prob - 1.0) < 0.01
        
        # Risk level should be valid
        assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_predict_high_risk(self, sample_employee_high_risk):
        """Test prediction for high-risk employee."""
        response = client.post("/predict", json=sample_employee_high_risk)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability_leave" in data
        
        # High-risk employee should have higher leave probability
        assert data["probability_leave"] > 0.3
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid data."""
        invalid_data = {
            "Age": 150,  # Invalid: too high
            "BusinessTravel": "InvalidValue"  # Invalid: not in enum
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self):
        """Test prediction with missing required fields."""
        incomplete_data = {
            "Age": 30,
            "BusinessTravel": "Travel_Rarely"
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint returns correct structure."""
        response = client.get("/model-info")
        
        # May return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "n_estimators" in data
            assert "max_depth" in data
            assert "feature_count" in data
            assert "features" in data
            assert isinstance(data["features"], list)
            assert len(data["features"]) > 0


class TestEdgeCases:
    """Edge case tests."""
    
    def test_boundary_values(self):
        """Test with boundary values for numeric fields."""
        boundary_data = {
            "Age": 18,
            "BusinessTravel": "Non-Travel",
            "DailyRate": 100,
            "Department": "Human Resources",
            "DistanceFromHome": 1,
            "Education": 1,
            "EducationField": "Human Resources",
            "EnvironmentSatisfaction": 1,
            "Gender": "Female",
            "HourlyRate": 30,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobRole": "Human Resources",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 1000,
            "MonthlyRate": 1000,
            "NumCompaniesWorked": 0,
            "OverTime": "No",
            "PercentSalaryHike": 0,
            "PerformanceRating": 1,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 0,
            "TrainingTimesLastYear": 0,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 0,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0
        }
        
        response = client.post("/predict", json=boundary_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability_stay" in data
        assert "probability_leave" in data
