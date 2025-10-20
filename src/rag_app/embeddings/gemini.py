from rag_app.embeddings.base import BaseEmbeddingModel

class GeminiEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, _client):
        self._client = _client

    def embed_document(self, document):
        response = self._client.models.embed_content(
            model="gemini-1.5-embedding",
            contents=document
        )
        return response.embeddings
    

    def embed_documents(self, documents):
        response = self._client.models.embed_content(
            model="gemini-1.5-embedding",
            contents=documents
        )
        return [embedding for embedding in response.embeddings]