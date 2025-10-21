from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_document(self, document):
        pass

    @abstractmethod
    async def embed_documents(self, documents):
        pass