"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Database
    database_path: str = "data/memory.db"
    vector_db_path: str = "data/vectors.db"

    # Embeddings
    embedding_provider: str = "openai"  # "openai", "local", "none"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Rate limiting / Pricing
    free_tier_limit: int = 100      # requests per day
    basic_tier_limit: int = 10000   # requests per month

    # Admin
    admin_api_key: str = "admin-secret-change-me"

    # Memory compression
    compression_threshold: int = 50   # max memories before triggering compression
    compression_batch_size: int = 10  # how many to summarise at once

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Ensure data directories exist
Path("data").mkdir(exist_ok=True)
