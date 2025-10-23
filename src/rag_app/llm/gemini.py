from google import genai
from pydantic import BaseModel
from .base import BaseLLMModel
from typing import List, Dict, Any, AsyncGenerator
from rag_app.config.settings import settings
from rag_app.config.logging import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class GeminiLLM(BaseLLMModel):
    def __init__(self, client):
        self._client: genai.Client = client
        self._model = settings.gemini.completion_model

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
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[prompt],
        )
        return response.text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM call, attempt {retry_state.attempt_number}")
    )
    async def generate_completion(self, messages: List[Dict[str, Any]], structured_format: BaseModel, temperature: float = 0.7) :
        # Convert messages to Gemini format
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[prompt],
            config={
            "response_mime_type": "application/json",
            "response_schema": structured_format,
        },
    )


        return response.parsed

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(f"Retrying LLM stream call, attempt {retry_state.attempt_number}")
    )
    async def stream_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=[prompt]
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text