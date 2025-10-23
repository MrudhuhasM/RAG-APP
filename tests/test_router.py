import pytest
from unittest.mock import AsyncMock, MagicMock
from rag_app.services.router import QueryRouter


@pytest.fixture
def mock_llm_model():
    return AsyncMock()


@pytest.fixture
def router(mock_llm_model):
    return QueryRouter(llm_model=mock_llm_model)


class TestQueryRouter:

    @pytest.mark.asyncio
    async def test_classify_query_simple(self, router, mock_llm_model):
        mock_llm_model.generate_response.return_value = "SIMPLE"
        result = await router._classify_query("What is AI?")
        assert result == False  # False for simple

    @pytest.mark.asyncio
    async def test_classify_query_complex(self, router, mock_llm_model):
        mock_llm_model.generate_response.return_value = "COMPLEX"
        result = await router._classify_query("Compare AI and ML")
        assert result == True  # True for complex

    @pytest.mark.asyncio
    async def test_classify_query_failure(self, router, mock_llm_model):
        mock_llm_model.generate_response.side_effect = Exception("Error")
        result = await router._classify_query("Test query")
        assert result == False  # Default to simple on error

    @pytest.mark.asyncio
    async def test_route_query_simple(self, router, mock_llm_model):
        mock_llm_model.generate_response.return_value = "SIMPLE"
        result = await router.route_query("What is AI?")
        assert result == "local"

    @pytest.mark.asyncio
    async def test_route_query_complex(self, router, mock_llm_model):
        mock_llm_model.generate_response.return_value = "COMPLEX"
        result = await router.route_query("Compare AI and ML")
        assert result == "gemini"