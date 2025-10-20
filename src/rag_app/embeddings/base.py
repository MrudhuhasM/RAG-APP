from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_document(self, document):
        pass

    @abstractmethod
    def embed_documents(self, documents):
        pass