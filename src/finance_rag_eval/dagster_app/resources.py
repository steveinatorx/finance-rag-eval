"""Dagster resources for configuration and paths."""

from pathlib import Path
from typing import Optional

from dagster import ConfigurableResource


class PathsResource(ConfigurableResource):
    """Resource for managing file paths."""

    outputs_dir: str = "outputs"
    sample_docs_dir: str = "src/finance_rag_eval/data/sample_docs"
    gold_set_path: str = "src/finance_rag_eval/data/qa_gold.json"

    def get_outputs_dir(self) -> Path:
        """Get outputs directory path."""
        return Path(self.outputs_dir)

    def get_sample_docs_dir(self) -> Path:
        """Get sample docs directory path."""
        return Path(self.sample_docs_dir)

    def get_gold_set_path(self) -> Path:
        """Get gold set path."""
        return Path(self.gold_set_path)


class LLMResource(ConfigurableResource):
    """Optional LLM resource (gated by API key)."""

    openai_api_key: Optional[str] = None

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.openai_api_key is not None and self.openai_api_key != ""
