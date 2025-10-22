from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

class OpenAISettings(BaseSettings):
    api_key: str = Field(default="")  # Will be validated
    embedding_model: str = "text-embedding-ada-002"
    completion_model: str = "gpt-3.5-turbo"

    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("OPENAI_API_KEY must be set in environment")
        return v


class GeminiSettings(BaseSettings):
    api_key: str = Field(default="")  # Will be validated
    embedding_model: str = "models/text-embedding-004"
    completion_model: str = "gemini-1.5-flash"

    model_config = SettingsConfigDict(env_prefix="GEMINI_")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GEMINI_API_KEY must be set in environment")
        return v

class LocalModelsSettings(BaseSettings):
    completion_base_url : str = "http://localhost:8080/v1"
    embedding_base_url : str = "http://localhost:8081/v1"

    model_config = SettingsConfigDict(env_prefix="LOCAL_MODELS_")

class EmbeddingSettings(BaseSettings):
    provider: str = "openai"  # Options: openai, gemini, local_models
    dimension: int = 768

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class LLMSettings(BaseSettings):
    provider: str = "openai"  # Options: openai, gemini, local

    model_config = SettingsConfigDict(env_prefix="LLM_")

class PineconeSettings(BaseSettings):
    api_key: str = Field(default="")  # Will be validated
    index_name: str = "rag-index"
    dimension: int = 1536
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    environment: str = "dev"
    questions_namespace: str = "questions"

    model_config = SettingsConfigDict(env_prefix="PINECONE_")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("PINECONE_API_KEY must be set in environment")
        return v


class RerankerSettings(BaseSettings):
    model: str = "BAAI/bge-reranker-base"

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl_seconds: Optional[int] = None  # Optional TTL for cached items
    embedding_ttl_seconds: Optional[int] = None  # Optional TTL for embeddings

    model_config = SettingsConfigDict(env_prefix="REDIS_")


class Settings(BaseSettings):

    env: str = "development"

    # Logging
    log_level: str = "INFO"

    # Application
    app_name: str = "RAG App"
    app_version: str = "0.1.0"
    app_description: str = "An API for Retrieval-Augmented Generation (RAG) applications."

    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    cors_allowed_origins: str = "*"  # Comma-separated in production
    
    # HTTP Client Timeouts
    http_timeout: float = 30.0
    embedding_timeout: float = 20.0

    # Nested settings - Pydantic will auto-populate from env vars
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    local_models: LocalModelsSettings = Field(default_factory=LocalModelsSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

# Create a global settings instance
settings = Settings()