import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from rag_app.app import app
from rag_app.services.rag import RagService
from rag_app.api.routes.rag import get_rag_service
import json

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    mock_service = MagicMock()
    app.dependency_overrides[get_rag_service] = lambda: mock_service
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_rag_service():
    service = MagicMock(spec=RagService)
    # Mock the async generator
    async def mock_answer_query(query):
        yield {"type": "token", "data": "Test answer"}
        yield {"type": "sources", "data": [
            {"content": "content1", "score": 0.9, "source": "src1", "rerank_score": 0.8},
            {"content": "content2", "score": 0.8, "source": "src2", "rerank_score": 0.7}
        ]}
    service.answer_query = mock_answer_query
    return service

class TestRAGRoutes:

    def test_query_success(self, mock_rag_service):
        # Override the dependency
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        
        response = client.post("/api/v1/query", json={"query": "What is AI?"})
        
        assert response.status_code == 200
        # Parse SSE response
        lines = response.text.strip().split('\n\n')
        events = []
        for line in lines:
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: '
                events.append(json.loads(data))
        
        assert len(events) == 2
        assert events[0]["type"] == "token"
        assert events[0]["data"] == "Test answer"
        assert events[1]["type"] == "sources"
        assert len(events[1]["data"]) == 2
        assert events[1]["data"][0]["content"] == "content1"
        
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
        async def failing_answer_query(query):
            raise Exception("Service error")
        mock_rag_service.answer_query = failing_answer_query
        
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        
        response = client.post("/api/v1/query", json={"query": "test"})
        
        assert response.status_code == 200  # Streaming response
        lines = response.text.strip().split('\n\n')
        events = []
        for line in lines:
            if line.startswith('data: '):
                data = line[6:]
                events.append(json.loads(data))
        
        assert any(e["type"] == "error" for e in events)
        
        app.dependency_overrides = {}

    def test_query_invalid_json(self):
        response = client.post("/api/v1/query", json={"invalid": "data"})
        
        assert response.status_code == 422  # Validation error