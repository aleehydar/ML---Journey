import pytest
from fastapi.testclient import TestClient

# Must mock db or auth easily here if needed, but for now we import the main app
from app import app
from cache.semantic_cache import semantic_cache

@pytest.fixture(scope="session")
def test_client():
    """Provides a reusable FastAPI TestClient."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def clear_cache():
    """Clears the semantic cache before every test run for deterministic behavior."""
    semantic_cache.cache.clear()
    yield
    semantic_cache.cache.clear()

@pytest.fixture
def auth_headers():
    """Provides default tenant dummy JWT or Auth headers if required by auth_middleware."""
    # Assuming the local `auth_middleware` accepts a bypass or a specific testing token format
    # This is a placeholder since the exact auth implementation might differ
    return {"Authorization": "Bearer TEST_TOKEN_PUBLIC_ORG"}
