import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rag_app.services.rag import RagService


@pytest.fixture
def mock_embed_model():
    return AsyncMock()


@pytest.fixture
def mock_vector_client():
    return AsyncMock()


@pytest.fixture
def mock_llm_model():
    return AsyncMock()


@pytest.fixture
def mock_encoder():
    encoder = MagicMock()
    encoder.predict.return_value = [0.9, 0.8]
    return encoder


@pytest.fixture
def rag_service(mock_embed_model, mock_vector_client, mock_llm_model, mock_encoder):
    with patch('rag_app.services.rag.settings') as mock_settings:
        mock_settings.reranker.model = "test_model"
        mock_settings.openai.completion_model = "gpt-3.5-turbo"
        mock_settings.pinecone.questions_namespace = "questions"
        return RagService(
            embed_model=mock_embed_model,
            vector_client=mock_vector_client,
            llm_model=mock_llm_model,
            encoder_model=mock_encoder
        )


class TestRagService:

    @pytest.mark.asyncio
    async def test_rerank_success(self, rag_service, mock_encoder):
        contexts = [{"content": "test1"}, {"content": "test2"}]
        result = await rag_service.rerank("query", contexts, top_k=2)
        assert len(result) == 2
        assert "rerank_score" in result[0]
        mock_encoder.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_failure(self, rag_service, mock_encoder):
        mock_encoder.predict.side_effect = Exception("Error")
        contexts = [{"content": "test"}]
        result = await rag_service.rerank("query", contexts)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_query_rewriter_success(self, rag_service, mock_llm_model):
        mock_llm_model.generate_completion.return_value = "Rewritten query"
        result = await rag_service.query_rewriter("original query")
        assert result == "Rewritten query"
        mock_llm_model.generate_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_rewriter_failure(self, rag_service, mock_llm_model):
        mock_llm_model.generate_completion.side_effect = Exception("Error")
        result = await rag_service.query_rewriter("query")
        assert result == "query"

    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, rag_service, mock_vector_client):
        mock_vector_client.query.return_value = {"matches": [{"id": "1", "score": 0.9}]}
        result = await rag_service.retrieve_documents([0.1, 0.2])
        assert len(result) == 1
        mock_vector_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_documents_failure(self, rag_service, mock_vector_client):
        mock_vector_client.query.side_effect = Exception("Error")
        result = await rag_service.retrieve_documents([0.1])
        assert result == []

    def test_format_results(self, rag_service):
        results = [{"metadata": {"node_content": "content", "source": "src"}, "score": 0.9}]
        formatted = rag_service.format_results(results)
        assert len(formatted) == 1
        assert formatted[0]["content"] == "content"

    def test_format_results_empty(self, rag_service):
        formatted = rag_service.format_results([])
        assert formatted == []

    def test_remove_duplicates(self, rag_service):
        results = [{"content": "dup"}, {"content": "dup"}, {"content": "unique"}]
        unique = rag_service.remove_duplicates(results)
        assert len(unique) == 2

    def test_remove_duplicates_empty(self, rag_service):
        unique = rag_service.remove_duplicates([])
        assert unique == []

    @pytest.mark.asyncio
    async def test_generate_answer_success(self, rag_service, mock_llm_model):
        contexts = [{"source": "src", "content": "content"}]
        mock_llm_model.generate_completion.return_value = "Answer"
        result = await rag_service.generate_answer("query", contexts)
        assert result == "Answer"

    @pytest.mark.asyncio
    async def test_generate_answer_no_contexts(self, rag_service):
        result = await rag_service.generate_answer("query", [])
        assert result == "No contexts available to generate an answer."

    @pytest.mark.asyncio
    async def test_generate_answer_no_query(self, rag_service):
        result = await rag_service.generate_answer("", [{"content": "c"}])
        assert result == "Please provide a valid query."

    @pytest.mark.asyncio
    async def test_generate_answer_failure(self, rag_service, mock_llm_model):
        contexts = [{"source": "src", "content": "content"}]
        mock_llm_model.generate_completion.side_effect = Exception("Error")
        result = await rag_service.generate_answer("query", contexts)
        assert result == "An error occurred while generating the answer."

    @pytest.mark.asyncio
    async def test_answer_query_success(self, rag_service, mock_embed_model, mock_vector_client, mock_llm_model, mock_encoder):
        mock_llm_model.generate_completion.side_effect = ["Rewritten", "Answer"]
        mock_embed_model.embed_document.return_value = [0.1]
        mock_vector_client.query.side_effect = [
            {"matches": [{"metadata": {"node_content": "content", "source": "src"}, "score": 0.9}]},
            {"matches": [{"metadata": {"node_content": "q_content", "source": "q_src"}, "score": 0.8}]}
        ]
        result = await rag_service.answer_query("query")
        assert "answer" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_answer_query_no_query(self, rag_service):
        result = await rag_service.answer_query("")
        assert result == {"answer": "Please provide a valid query.", "sources": []}

    @pytest.mark.asyncio
    async def test_answer_query_failure(self, rag_service, mock_embed_model):
        mock_embed_model.embed_document.side_effect = Exception("Error")
        result = await rag_service.answer_query("query")
        assert result == {"answer": "An error occurred while processing your query.", "sources": []}