from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    api_key: str = "your-openai-api-key"
    embedding_model : str = "text-embedding-ada-002"
    completion_model : str = "gpt-3.5-turbo"

    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class GeminiSettings(BaseSettings):
    api_key: str = "your-google-api-key"
    embedding_model : str = "text-embedding-ada-002"
    completion_model : str = "gpt-3.5-turbo"

    model_config = SettingsConfigDict(env_prefix="GEMINI_")

class LocalModelsSettings(BaseSettings):
    completion_base_url : str = "http://localhost:8080/v1"
    embedding_base_url : str = "http://localhost:8081/v1"

    model_config = SettingsConfigDict(env_prefix="LOCAL_MODELS_")
    

class Settings(BaseSettings):

    env : str = "development"

    # Logging
    log_level: str = "INFO"

    # Application
    app_name: str = "RAG App"
    app_version: str = "0.1.0"
    app_description: str = "An API for Retrieval-Augmented Generation (RAG) applications."

    openai: OpenAISettings = OpenAISettings()
    gemini: GeminiSettings = GeminiSettings()
    local_models: LocalModelsSettings = LocalModelsSettings()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global settings instance
settings = Settings()