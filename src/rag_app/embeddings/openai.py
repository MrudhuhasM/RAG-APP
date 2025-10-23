import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rag_app.embeddings.base import BaseEmbeddingModel
from rag_app.config.settings import settings
from rag_app.config.logging import logger


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, _client, _model_name: str):
        self._client = _client
        self._model = _model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying embedding call, attempt {retry_state.attempt_number}")
    )
    async def embed_document(self, document):
        response = await self._client.embeddings.create(
            model=self._model,
            input=document
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying batch embedding call, attempt {retry_state.attempt_number}")
    )
    async def embed_documents(self, documents):
        response = await self._client.embeddings.create(
            model=self._model,
            input=documents
        )
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: [item.embedding for item in response.data]
        )
        return results

