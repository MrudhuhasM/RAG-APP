import asyncio
from rag_app.config.settings import settings
from google.genai import types
from rag_app.embeddings.base import BaseEmbeddingModel

class GeminiEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, _client):
        self._client = _client

    async def embed_document(self, document):
        response = await self._client.aio.models.embed_content(
            model=settings.gemini.embedding_model,
            contents=document,
            config=types.EmbedContentConfig(output_dimensionality=settings.embedding.dimension)
        )
        return response.embeddings[0].values
    

    async def embed_documents(self, documents):
        response = await self._client.aio.models.embed_content(
            model=settings.gemini.embedding_model,
            contents=documents,
            config=types.EmbedContentConfig(output_dimensionality=settings.embedding.dimension)
        )

        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: [embedding for embedding in response.embeddings[0].values]
        )
        return results