from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # Alias allows pydantic to pick up DATABASE_URL from .env automatically
    database_url_env: Optional[str] = Field(None, alias="DATABASE_URL")
    
    # Critical variables should not have "working" defaults in production
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    
    # AI Provider Configuration (Qwen via Alibaba Cloud DashScope, OpenAI-compatible endpoint).
    # Swapping to a self-hosted vLLM/Ollama server later only requires changing ai_base_url
    # (and ai_api_key to any non-empty placeholder) — no code changes.
    ai_api_key: str = Field(..., alias="AI_API_KEY")
    ai_base_url: str = Field("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", alias="AI_BASE_URL")
    model_chat: str = Field("qwen-max", alias="MODEL_CHAT")
    model_chat_light: str = Field("qwen-flash", alias="MODEL_CHAT_LIGHT")
    model_vision: str = Field("qwen-vl-max", alias="MODEL_VISION")
    model_embedding: str = Field("text-embedding-v4", alias="MODEL_EMBEDDING")
    embedding_dim: int = Field(2048, alias="EMBEDDING_DIM")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    # Optional Service Overrides
    api_url: str = Field("http://127.0.0.1:8000", alias="API_URL")
    frontend_url: str = Field("http://localhost:3000", alias="FRONTEND_URL")
    
    # Cloudinary Config (added 2026-04-12)
    cloudinary_cloud_name: Optional[str] = Field(None, alias="CLOUDINARY_CLOUD_NAME")
    cloudinary_api_key: Optional[str] = Field(None, alias="CLOUDINARY_API_KEY")
    cloudinary_api_secret: Optional[str] = Field(None, alias="CLOUDINARY_API_SECRET")
    
    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        if all([self.postgres_user, self.postgres_password, self.postgres_host, self.postgres_db]):
            return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        return "" # Fallback to empty if not configured

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        populate_by_name = True

settings = Settings()
