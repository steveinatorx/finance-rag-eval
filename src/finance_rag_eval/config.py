"""Configuration management with environment variables and optional YAML."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # API Keys (optional)
    openai_api_key: Optional[str] = None
    openai_project_id: Optional[str] = None  # Optional: for project-scoped keys
    sec_edgar_api_key: Optional[str] = None

    # Model Configuration
    embedding_model: str = "sentence-transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model: Optional[str] = None
    llm_temperature: float = 0.0

    # Paths
    outputs_dir: Path = Path("outputs")
    sample_docs_dir: Path = Path("src/finance_rag_eval/data/sample_docs")

    class Config:
        # Note: Uses environment variables (set via .envrc with direnv, or manually)
        # Also checks .env file for compatibility
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
