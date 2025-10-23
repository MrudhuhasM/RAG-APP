from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.embeddings import BaseEmbedding

from rag_app.embeddings.openai import OpenAIEmbeddingModel
from rag_app.embeddings.gemini import GeminiEmbeddingModel

from rag_app.config.settings import settings
from rag_app.config.logging import logger
from rag_app.embeddings.base import BaseEmbeddingModel

from openai import OpenAI, AsyncOpenAI
from google import genai

def get_chunk_embeddings() -> BaseEmbedding:
    if settings.embedding.provider == "openai":
        logger.info("Using llama-index OpenAI Embedding Model")
        return OpenAIEmbedding(model=settings.openai.embedding_model, api_key=settings.openai.api_key, embed_batch_size=32)
    elif settings.embedding.provider == "gemini":
        logger.info("Using llama-index Gemini Embedding Model")
        return GeminiEmbedding(model=settings.gemini.embedding_model, api_key=settings.gemini.api_key, embed_batch_size=32)
    elif settings.embedding.provider == "local":
        logger.info("Using llama-index Local Embedding Model")
        return OpenAIEmbedding(api_base=settings.local_models.embedding_base_url, model="text-embedding-3-small", embed_batch_size=32)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")
    

def get_embed_model() -> BaseEmbeddingModel:
    if settings.embedding.provider == "openai":
        logger.info("Initializing OpenAI Embedding Model")
        client = AsyncOpenAI(api_key=settings.openai.api_key, timeout=settings.embedding_timeout)
        return OpenAIEmbeddingModel(_client=client, _model_name=settings.openai.embedding_model)
    elif settings.embedding.provider == "gemini":
        logger.info("Initializing Gemini Embedding Model")
        client = genai.Client(api_key=settings.gemini.api_key)
        return GeminiEmbeddingModel(_client=client, _model_name=settings.gemini.embedding_model)
    elif settings.embedding.provider == "local":
        logger.info("Initializing Local Models Embedding Model")
        client = AsyncOpenAI(base_url=settings.local_models.embedding_base_url, api_key="test", timeout=settings.embedding_timeout)
        return OpenAIEmbeddingModel(_client=client, _model_name=settings.local_models.embedding_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")

