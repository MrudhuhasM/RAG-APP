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

    embedding_model: str = "nomic-embed-text-v2-moe-q8_0.gguf"
    completion_model: str = "Qwen/Qwen3-0.6B"

    model_config = SettingsConfigDict(env_prefix="LOCAL_MODELS_")

class EmbeddingSettings(BaseSettings):
    provider: str = "openai"  # Options: openai, gemini, local_models
    dimension: int = 1024

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class LLMSettings(BaseSettings):
    provider: str = "openai"  # Options: openai, gemini, local

    model_config = SettingsConfigDict(env_prefix="LLM_")

class PineconeSettings(BaseSettings):
    api_key: str = Field(default="")  # Will be validated
    index_name: str = "rag-index"
    dimension: int = 1024
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    environment: str = "dev"
    questions_namespace: str = "questions"
    semantic_cache_namespace: str = "semantic-cache"

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

    semantic_threshold: float = Field(default=0.85, description="Threshold for semantic similarity")

    # Nested settings - Pydantic will auto-populate from env vars
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    local_models: LocalModelsSettings = Field(default_factory=LocalModelsSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    MODEL_COSTS: dict[str, dict[str, float]] = {
  # --- OpenAI ---
  "gpt-5":                 {"input": 1.25,  "output": 10.00},
  "gpt-5-mini":            {"input": 0.25,  "output": 2.00},
  "gpt-5-nano":            {"input": 0.05,  "output": 0.40},
  "gpt-5-pro":             {"input": 15.00, "output": 120.00},
  "gpt-realtime":          {"input": 4.00,  "output": 16.00},
  "gpt-realtime-mini":     {"input": 0.60,  "output": 2.40},

  # --- Google Gemini (AI Studio) ---
  "gemini-2.5-pro (<=200k prompt)": {"input": 1.25, "output": 10.00},   # per 1M tokens
  "gemini-2.5-pro (>200k prompt)":  {"input": 2.50, "output": 15.00},
  "gemini-2.5-flash":               {"input": 0.30, "output": 2.50},
  "gemini-2.5-flash-lite":          {"input": 0.10, "output": 0.40},
  "gemini-2.0-flash":               {"input": 0.10, "output": 0.40},
  "gemini-2.0-flash-lite":          {"input": 0.075,"output": 0.30},

  # --- Gemma (Google AI Studio) ---
  # Gemma is free-of-charge on AI Studio at the moment (no paid tier pricing shown).
  "gemma-3":                        {"input": 0.00, "output": 0.00},    # Free tier only
  "gemma-3n":                       {"input": 0.00, "output": 0.00},    # Free tier only
  "gemma-3-27b-it":                 {"input": 0.25, "output": 0.75},    # Free tier only Testing per 1M input/output tokens

  # --- Local Models ---
  "gaunernst/gemma-3-1b-it-int4-awq":                 {"input": 0.10, "output": 0.50}, # Testing per 1M input/output tokens

  # --- Embedding Models ---
  # OpenAI
  "text-embedding-3-small":         {"input": 0.02, "output": 0.00},    # per 1M input tokens
  "Qwen/Qwen3-0.6B-embedding":      {"input": 0.01, "output": 0.00},    # Local model - no cost
  # Google (Gemini API)
  "gemini-embedding-001":           {"input": 0.15, "output": 0.00},    # Standard, per 1M input tokens
  "gemini-embedding-001 (batch)":   {"input": 0.075,"output": 0.00}     # Batch, per 1M input tokens
}


    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

# Create a global settings instance
settings = Settings()