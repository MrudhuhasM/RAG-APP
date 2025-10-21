import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from llama_index.core import Document
from llama_index.core.schema import TextNode
from rag_app.services.ingest import IngestionService


@pytest.fixture
def mock_reader():
    reader = MagicMock()
    reader.load_data = MagicMock(return_value=[])
    return reader


@pytest.fixture
def mock_node_parser():
    parser = MagicMock()
    parser.get_nodes_from_documents = MagicMock(return_value=[])
    return parser


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    return client


@pytest.fixture
def mock_vector_client():
    client = AsyncMock()
    return client


@pytest.fixture
def ingestion_service(mock_reader, mock_node_parser, mock_llm_client, mock_embedding_client, mock_vector_client):
    return IngestionService(
        reader=mock_reader,
        node_parser=mock_node_parser,
        client=mock_llm_client,
        embedding_client=mock_embedding_client,
        vector_client=mock_vector_client
    )


class TestIngestionService:

    @pytest.mark.asyncio
    async def test_ingest_file_success(self, ingestion_service, mock_reader):
        mock_reader.load_data.return_value = [Document(text="Test content")]
        documents = await ingestion_service._ingest_file("test.pdf")
        assert len(documents) == 1
        assert documents[0].get_content() == "Test content"

    @pytest.mark.asyncio
    async def test_ingest_file_failure(self, ingestion_service, mock_reader):
        mock_reader.load_data.side_effect = Exception("Load error")
        documents = await ingestion_service._ingest_file("test.pdf")
        assert documents == []

    @pytest.mark.asyncio
    async def test_filter_empty_documents(self, ingestion_service):
        docs = [
            Document(text="Content"),
            Document(text=""),
            Document(text="   "),
            Document(text="More content")
        ]
        filtered = await ingestion_service._filter_empty_documents(docs)
        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_preprocess_documents(self, ingestion_service):
        docs = [Document(text="Content"), Document(text="")]
        preprocessed = await ingestion_service._preprocess_documents(docs)
        assert len(preprocessed) == 1

    @pytest.mark.asyncio
    async def test_chunk_documents(self, ingestion_service, mock_node_parser):
        docs = [Document(text="Content")]
        nodes = [TextNode(id_="1", text="Content")]
        mock_node_parser.get_nodes_from_documents.return_value = nodes
        chunked = await ingestion_service._chunk_documents(docs)
        assert len(chunked) == 1

    def test_build_prompt(self, ingestion_service):
        node = TextNode(id_="1", text="Test content")
        prompt = ingestion_service._build_prompt(node)
        assert "Test content" in prompt
        assert "questions" in prompt

    @pytest.mark.asyncio
    async def test_call_llm_success(self, ingestion_service, mock_llm_client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"questions": ["Q1"]}'
        mock_llm_client.chat.completions.create.return_value = mock_response
        response = await ingestion_service._call_llm("prompt")
        assert response == '{"questions": ["Q1"]}'

    @pytest.mark.asyncio
    async def test_call_llm_failure(self, ingestion_service, mock_llm_client):
        mock_llm_client.chat.completions.create.side_effect = Exception("API error")
        response = await ingestion_service._call_llm("prompt")
        assert response == ""

    def test_parse_llm_response_valid(self, ingestion_service):
        response = '{"questions": ["Q1", "Q2"]}'
        questions = ingestion_service._parse_llm_response(response)
        assert questions == ["Q1", "Q2"]

    def test_parse_llm_response_with_code_block(self, ingestion_service):
        response = '```json\n{"questions": ["Q1"]}\n```'
        questions = ingestion_service._parse_llm_response(response)
        assert questions == ["Q1"]

    def test_parse_llm_response_invalid(self, ingestion_service):
        with pytest.raises(json.JSONDecodeError):
            ingestion_service._parse_llm_response("invalid json")

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, ingestion_service, mock_llm_client):
        node = TextNode(id_="1", text="Content")
        mock_llm_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"questions": ["Q1"]}'))]
        )
        processed_node, success = await ingestion_service._extract_metadata(node)
        assert success
        assert processed_node.metadata["questions"] == ["Q1"]

    @pytest.mark.asyncio
    async def test_extract_metadata_failure(self, ingestion_service, mock_llm_client):
        node = TextNode(id_="1", text="Content")
        mock_llm_client.chat.completions.create.side_effect = Exception("Error")
        processed_node, success = await ingestion_service._extract_metadata(node)
        assert not success

    @pytest.mark.asyncio
    async def test_process_nodes_success(self, ingestion_service, mock_llm_client):
        nodes = [TextNode(id_="1", text="Content")]
        mock_llm_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"questions": ["Q1"]}'))]
        )
        processed = await ingestion_service._process_nodes(nodes)
        assert len(processed) == 1

    @pytest.mark.asyncio
    async def test_process_nodes_high_failure_rate(self, ingestion_service, mock_llm_client):
        nodes = [TextNode(id_="1", text="Content")] * 10
        mock_llm_client.chat.completions.create.side_effect = Exception("Error")
        with pytest.raises(Exception, match="High failure rate"):
            await ingestion_service._process_nodes(nodes)

    @pytest.mark.asyncio
    async def test_embed_node_success(self, ingestion_service, mock_embedding_client):
        node = TextNode(id_="1", text="Content")
        mock_embedding_client.embed_document.return_value = [0.1, 0.2]
        processed_node, success = await ingestion_service._embed_node(node)
        assert success
        assert processed_node.embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_node_failure(self, ingestion_service, mock_embedding_client):
        node = TextNode(id_="1", text="Content")
        mock_embedding_client.embed_document.side_effect = Exception("Embed error")
        processed_node, success = await ingestion_service._embed_node(node)
        assert not success

    @pytest.mark.asyncio
    async def test_embed_nodes_success(self, ingestion_service, mock_embedding_client):
        nodes = [TextNode(id_="1", text="Content")]
        mock_embedding_client.embed_document.return_value = [0.1, 0.2]
        embedded = await ingestion_service._embed_nodes(nodes)
        assert len(embedded) == 1

    @pytest.mark.asyncio
    async def test_embed_nodes_high_failure_rate(self, ingestion_service, mock_embedding_client):
        nodes = [TextNode(id_="1", text="Content")] * 10
        mock_embedding_client.embed_document.side_effect = Exception("Error")
        with pytest.raises(Exception, match="High failure rate"):
            await ingestion_service._embed_nodes(nodes)

    def test_node_to_vector(self, ingestion_service):
        node = TextNode(id_="1", text="Content")
        node.embedding = [0.1]
        node.metadata = {"key": "value"}
        vector = ingestion_service._node_to_vector(node)
        assert vector["id"] == "1"
        assert vector["values"] == [0.1]
        assert vector["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_upsert_nodes(self, ingestion_service, mock_vector_client):
        node = TextNode(id_="1", text="Content")
        node.embedding = [0.1]
        node.metadata = {}
        nodes = [node]
        await ingestion_service._upsert_nodes(nodes)
        mock_vector_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_question_to_vector(self, ingestion_service, mock_embedding_client):
        node = TextNode(id_="1", text="Content")
        node.metadata = {"source": "src"}
        mock_embedding_client.embed_document.return_value = [0.1]
        vector = await ingestion_service._question_to_vector("Question", node, 0)
        assert vector["id"] == "1_q_0"
        assert vector["values"] == [0.1]

    @pytest.mark.asyncio
    async def test_upsert_questions(self, ingestion_service, mock_vector_client, mock_embedding_client):
        node = TextNode(id_="1", text="Content")
        node.metadata = {"questions": ["Q1"]}
        nodes = [node]
        mock_embedding_client.embed_document.return_value = [0.1]
        await ingestion_service._upsert_questions(nodes)
        mock_vector_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_full_flow(self, ingestion_service, mock_reader, mock_node_parser, mock_llm_client, mock_embedding_client, mock_vector_client):
        # Mock all steps
        mock_reader.load_data.return_value = [Document(text="Content")]
        mock_node_parser.get_nodes_from_documents.return_value = [TextNode(id_="1", text="Content")]
        mock_llm_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"questions": ["Q1"]}'))]
        )
        mock_embedding_client.embed_document.return_value = [0.1]
        mock_vector_client.upsert = AsyncMock()

        nodes = await ingestion_service.ingest("test.pdf")
        assert len(nodes) == 1
        # Verify calls
        mock_reader.load_data.assert_called_once_with("test.pdf")
        mock_node_parser.get_nodes_from_documents.assert_called_once()
        mock_llm_client.chat.completions.create.assert_called_once()
        mock_embedding_client.embed_document.assert_called()
        mock_vector_client.upsert.assert_called()