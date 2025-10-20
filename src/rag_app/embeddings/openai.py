from rag_app.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, _client):
        self._client = _client

    def embed_document(self, document):
        response = self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=document
        )
        return response.data[0].embedding
    
    def embed_documents(self, documents):
        response = self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=documents
        )
        return [item.embedding for item in response.data]
    
    