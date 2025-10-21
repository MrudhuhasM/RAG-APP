from rag_app.llm.base import BaseLLMModel
from rag_app.config.settings import settings
from rag_app.config.logging import logger

def get_llm_model() -> BaseLLMModel:
    if settings.llm.provider == "openai":
        from rag_app.llm.openai import OpenAILLM
        logger.info("Initializing OpenAI LLM")
        return OpenAILLM(api_key=settings.openai.api_key)
    elif settings.llm.provider == "gemini":
        from rag_app.llm.gemini import GeminiLLM
        logger.info("Initializing Gemini LLM")
        return GeminiLLM(api_key=settings.gemini.api_key,model=settings.gemini.completion_model)
    elif settings.llm.provider == "local":
        from rag_app.llm.local import LocalLLM
        logger.info("Initializing Local LLM")
        return LocalLLM(base_url=settings.local_models.completion_base_url)
    else:
        logger.error(f"Unsupported LLM provider: {settings.llm.provider}")
        raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")