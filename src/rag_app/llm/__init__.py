from rag_app.llm.base import BaseLLMModel
from rag_app.config.settings import settings
from rag_app.config.logging import logger
from google import genai


def get_llm_model_by_provider(provider: str) -> BaseLLMModel:
    """
    Get an LLM model instance for a specific provider.
    
    Args:
        provider: The provider name ("openai", "gemini", or "local")
        
    Returns:
        BaseLLMModel: An instance of the appropriate LLM model
    """
    if provider == "openai":
        from rag_app.llm.openai import OpenAILLM
        logger.info(f"Initializing OpenAI LLM for provider: {provider}")
        return OpenAILLM(api_key=settings.openai.api_key)
    elif provider == "gemini":
        from rag_app.llm.gemini import GeminiLLM
        logger.info(f"Initializing Gemini LLM for provider: {provider}")
        client = genai.Client(api_key=settings.gemini.api_key)
        return GeminiLLM(client=client)
    elif provider == "local":
        from rag_app.llm.local import LocalLLM
        logger.info(f"Initializing Local LLM for provider: {provider}")
        return LocalLLM(base_url=settings.local_models.completion_base_url)
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm_model() -> BaseLLMModel:
    """
    Get the default LLM model based on settings.
    
    Returns:
        BaseLLMModel: An instance of the configured default LLM model
    """
    return get_llm_model_by_provider(settings.llm.provider)