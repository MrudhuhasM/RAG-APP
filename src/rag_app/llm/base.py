from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator


class BaseLLMModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    async def generate_completion(self, messages: List[Dict[str, Any]], structured_format: BaseModel, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    async def stream_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        pass