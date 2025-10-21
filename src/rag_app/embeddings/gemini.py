import asyncio
from rag_app.embeddings.base import BaseEmbeddingModel

class GeminiEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, _client):
        self._client = _client

    async def embed_document(self, document):
        response = await self._client.aio.models.embed_content(
            model="gemini-1.5-embedding",
            contents=document
        )
        return response.embeddings
    

    async def embed_documents(self, documents):
        response = await self._client.aio.models.embed_content(
            model="gemini-1.5-embedding",
            contents=documents
        )

        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: [embedding for embedding in response.embeddings]
        )
        return results