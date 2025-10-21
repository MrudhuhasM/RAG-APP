from openai import AsyncOpenAI
from .base import BaseLLMModel
from typing import List, Dict, Any


class LocalLLM(BaseLLMModel):
    def __init__(self, base_url: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")  # Local models may not require a key

    async def generate_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()