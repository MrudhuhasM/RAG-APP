from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator


class BaseLLMModel(ABC):
    @abstractmethod
    async def generate_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    async def stream_completion(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 500, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        pass