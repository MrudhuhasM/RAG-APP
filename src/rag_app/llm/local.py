from openai import AsyncOpenAI
from pydantic import BaseModel
from rag_app.config.logging import logger
from rag_app.config.settings import settings
from .base import BaseLLMModel
from typing import List, Dict, Any, AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LocalLLM(BaseLLMModel):
    def __init__(self, base_url: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")  # Local models may not require a key
        self._model = settings.local_models.completion_model

    @property
    def model_name(self) -> str:
        return self._model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call, attempt {retry_state.attempt_number}")
    )
    async def generate_response(self, messages: List[Dict[str, Any]], temperature: float = 0.7) :
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temperature
        )
        content = response.choices[0].message.content
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call, attempt {retry_state.attempt_number}")
    )
    async def generate_completion(self, messages: List[Dict[str, Any]], structured_format: BaseModel, temperature: float = 0.7):
        response = await self.client.chat.completions.parse(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temperature,
            response_format=structured_format
        )
        content = response.choices[0].message.parsed
        return content
    

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM stream call, attempt {retry_state.attempt_number}")
    )
    async def stream_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=temperature,
            stream=True
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content