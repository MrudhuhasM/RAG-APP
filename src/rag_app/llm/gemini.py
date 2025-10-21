from google import genai
from .base import BaseLLMModel
from typing import List, Dict, Any, AsyncGenerator
from rag_app.config.settings import settings


class GeminiLLM(BaseLLMModel):
    def __init__(self, client):
        self._client: genai.Client = client

    async def generate_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        # Convert messages to Gemini format
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self._client.aio.models.generate_content(
            model = settings.gemini.completion_model,
            contents= [prompt]
        )

        return response.text.strip() if response.text else ""
    
    async def stream_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self._client.aio.models.generate_content_stream(
            model= settings.gemini.completion_model,
            contents= [prompt]
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text