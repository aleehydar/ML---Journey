"""
Model Tests for Employee Attrition Prediction
"""

import pytest
import joblib
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
ENCODINGS_PATH = os.path.join(MODEL_DIR, "encodings.json")


class TestModelArtifacts:
    """Tests for model artifacts."""
    
    def test_model_file_exists(self):
        """Test that model file was created."""
        assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    
    def test_feature_columns_file_exists(self):
        """Test that feature columns file was created."""
        assert os.path.exists(FEATURE_COLUMNS_PATH), f"Feature columns file not found"
    
    def test_metrics_file_exists(self):
        """Test that metrics file was created."""
        assert os.path.exists(METRICS_PATH), f"Metrics file not found"
    
    def test_encodings_file_exists(self):
        """Test that encodings file was created."""
        assert os.path.exists(ENCODINGS_PATH), f"Encodings file not found"


class TestModelLoading:
    """Tests for loading the model."""
    
    @pytest.fixture
    def loaded_model(self):
        """Load the trained model."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model not trained yet")
        return joblib.load(MODEL_PATH)
    
    @pytest.fixture
    def feature_columns(self):
        """Load feature columns."""
        if not os.path.exists(FEATURE_COLUMNS_PATH):
            pytest.skip("Feature columns not available")
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            return json.load(f)
    
    def test_model_type(self, loaded_model):
        """Test that loaded model is RandomForestClassifier."""
        assert isinstance(loaded_model, RandomForestClassifier)
    
    def test_model_has_required_attributes(self, loaded_model):
        """Test that model has required attributes."""
        assert hasattr(loaded_model, 'n_estimators')
        assert hasattr(loaded_model, 'feature_importances_')
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'predict_proba')
    
    def test_feature_importances_shape(self, loaded_model, feature_columns):
        """Test that feature importances match number of features."""
        assert len(loaded_model.feature_importances_) == len(feature_columns)
    
    def test_model_can_predict(self, loaded_model, feature_columns):
        """Test that model can make predictions."""
        # Create dummy input
        dummy_input = np.zeros((1, len(feature_columns)))
        
        prediction = loaded_model.predict(dummy_input)
        probabilities = loaded_model.predict_proba(dummy_input)
        
        assert prediction.shape == (1,)
        assert probabilities.shape == (1, 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert abs(np.sum(probabilities) - 1.0) < 0.01


class TestMetrics:
    """Tests for model metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Load metrics."""
        if not os.path.exists(METRICS_PATH):
            pytest.skip("Metrics not available")
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    
    def test_metrics_structure(self, metrics):
        """Test that metrics has required fields."""
        required_fields = ['accuracy', 'roc_auc', 'training_samples', 'test_samples']
        for field in required_fields:
            assert field in metrics, f"Missing field: {field}"
    
    def test_accuracy_range(self, metrics):
        """Test that accuracy is in valid range."""
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_roc_auc_range(self, metrics):
        """Test that ROC-AUC is in valid range."""
        assert 0 <= metrics['roc_auc'] <= 1
        # ROC-AUC should be at least 0.5 (better than random)
        assert metrics['roc_auc'] >= 0.5
    
    def test_cv_scores_present(self, metrics):
        """Test that cross-validation scores are present."""
        assert 'cv_roc_auc_mean' in metrics
        assert 'cv_roc_auc_std' in metrics
        assert metrics['cv_roc_auc_mean'] >= 0.5


class TestFeatureColumns:
    """Tests for feature columns."""
    
    @pytest.fixture
    def feature_columns(self):
        """Load feature columns."""
        if not os.path.exists(FEATURE_COLUMNS_PATH):
            pytest.skip("Feature columns not available")
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            return json.load(f)
    
    def test_feature_columns_is_list(self, feature_columns):
        """Test that feature columns is a list."""
        assert isinstance(feature_columns, list)
    
    def test_feature_columns_not_empty(self, feature_columns):
        """Test that feature columns is not empty."""
        assert len(feature_columns) > 0
    
    def test_no_duplicate_features(self, feature_columns):
        """Test that there are no duplicate feature names."""
        assert len(feature_columns) == len(set(feature_columns))
    
    def test_all_features_are_strings(self, feature_columns):
        """Test that all feature names are strings."""
        assert all(isinstance(f, str) for f in feature_columns)


class TestEncodings:
    """Tests for categorical encodings."""
    
    @pytest.fixture
    def encodings(self):
        """Load encodings."""
        if not os.path.exists(ENCODINGS_PATH):
            pytest.skip("Encodings not available")
        with open(ENCODINGS_PATH, 'r') as f:
            return json.load(f)
    
    def test_encodings_is_dict(self, encodings):
        """Test that encodings is a dictionary."""
        assert isinstance(encodings, dict)
    
    def test_required_categorical_variables(self, encodings):
        """Test that required categorical variables are encoded."""
        required_vars = [
            'BusinessTravel',
            'Department',
            'Gender',
            'JobRole',
            'MaritalStatus',
            'OverTime'
        ]
        for var in required_vars:
            assert var in encodings, f"Missing encoding for {var}"
    
    def test_encoding_values_are_integers(self, encodings):
        """Test that all encoding values are integers."""
        for var, mapping in encodings.items():
            for value, code in mapping.items():
                assert isinstance(code, int)
