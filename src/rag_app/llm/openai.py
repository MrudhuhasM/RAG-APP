from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseLLMModel
from typing import List, Dict, Any
from rag_app.config.settings import settings
from rag_app.config.logging import logger


class OpenAILLM(BaseLLMModel):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, timeout=settings.http_timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call, attempt {retry_state.attempt_number}")
    )
    async def generate_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""