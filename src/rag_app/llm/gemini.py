import google.generativeai as genai
from .base import BaseLLMModel
from typing import List, Dict, Any


class GeminiLLM(BaseLLMModel):
    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    async def generate_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        # Convert messages to Gemini format
        prompt = "\n".join([msg["content"] for msg in messages])
        response = await self.model.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ))
        return response.text.strip()