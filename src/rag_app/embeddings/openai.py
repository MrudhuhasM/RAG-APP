import asyncio
from rag_app.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, _client):
        self._client = _client

    async def embed_document(self, document):
        response = await self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=document
        )
        return response.data[0].embedding

    async def embed_documents(self, documents):
        response = await self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=documents
        )
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: [item.embedding for item in response.data]
        )
        return results

