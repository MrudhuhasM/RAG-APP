import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from rag_app.app import app
from rag_app.services.rag import RagService
from rag_app.api.routes.rag import get_rag_service

client = TestClient(app)

@pytest.fixture
def mock_rag_service():
    service = MagicMock(spec=RagService)
    service.answer_query = AsyncMock(return_value={
        "answer": "Test answer",
        "sources": [
            {"content": "content1", "score": 0.9, "source": "src1", "rerank_score": 0.8},
            {"content": "content2", "score": 0.8, "source": "src2", "rerank_score": 0.7}
        ]
    })
    return service

class TestRAGRoutes:

    def test_query_success(self, mock_rag_service):
        # Override the dependency
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        
        response = client.post("/api/v1/query", json={"query": "What is AI?"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["content"] == "content1"
        mock_rag_service.answer_query.assert_called_once_with("What is AI?")
        
        # Clean up
        app.dependency_overrides = {}

    def test_query_empty_query(self):
        response = client.post("/api/v1/query", json={"query": ""})
        
        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    def test_query_whitespace_query(self):
        response = client.post("/api/v1/query", json={"query": "   "})
        
        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    def test_query_service_failure(self, mock_rag_service):
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        mock_rag_service.answer_query.side_effect = Exception("Service error")
        
        response = client.post("/api/v1/query", json={"query": "test"})
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
        
        app.dependency_overrides = {}

    def test_query_invalid_response(self, mock_rag_service):
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        mock_rag_service.answer_query.return_value = {"invalid": "response"}
        
        response = client.post("/api/v1/query", json={"query": "test"})
        
        assert response.status_code == 500
        assert "Invalid service response" in response.json()["detail"]
        
        app.dependency_overrides = {}

    def test_query_invalid_json(self):
        response = client.post("/api/v1/query", json={"invalid": "data"})
        
        assert response.status_code == 422  # Validation error

    def test_query_service_initialization_failure(self):
        # Mock the get_rag_service function to raise exception
        def failing_dependency():
            raise Exception("Init failed")
        
        app.dependency_overrides[get_rag_service] = failing_dependency
        
        with pytest.raises(Exception, match="Init failed"):
            client.post("/api/v1/query", json={"query": "test"})
        
        app.dependency_overrides = {}