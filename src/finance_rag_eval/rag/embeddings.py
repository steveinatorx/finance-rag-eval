"""Embedding generation: sentence-transformers (default) and optional OpenAI."""

from typing import List, Optional

import numpy as np

from finance_rag_eval.config import settings
from finance_rag_eval.constants import DEFAULT_EMBEDDING_MODEL
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)

# Lazy loading of sentence-transformers
_sentence_model = None


def _get_sentence_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Lazy load sentence-transformers model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {model_name}")
            _sentence_model = SentenceTransformer(model_name)
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
    return _sentence_model


def _get_openai_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings from OpenAI API."""
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key not set")

    try:
        from openai import OpenAI

        # Initialize client with optional project ID
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_project_id:
            client_kwargs["default_headers"] = {
                "OpenAI-Project": settings.openai_project_id
            }

        client = OpenAI(**client_kwargs)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except ImportError:
        raise ValueError("OpenAI library not installed")


def generate_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings
        model_name: Optional model name (defaults to config)

    Returns:
        numpy array of embeddings (n_texts, embedding_dim)
    """
    if settings.embedding_model == "openai" and settings.openai_api_key:
        logger.info("Using OpenAI embeddings")
        return _get_openai_embeddings(texts)
    else:
        # Default to sentence-transformers
        model_name = (
            model_name or settings.embedding_model_name or DEFAULT_EMBEDDING_MODEL
        )
        model = _get_sentence_model(model_name)
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
