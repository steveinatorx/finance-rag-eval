"""Dagster assets for embedding generation."""

from typing import List

import numpy as np
from dagster import AssetExecutionContext, asset

from finance_rag_eval.rag.embeddings import generate_embeddings


@asset(key_prefix=["rag"])
def embeddings(context: AssetExecutionContext, chunks: List[dict]) -> np.ndarray:
    """
    Generate embeddings for chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        numpy array of embeddings
    """
    context.log.info(f"Generating embeddings for {len(chunks)} chunks")

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings_array = generate_embeddings(chunk_texts)

    context.log.info(f"Generated embeddings with shape {embeddings_array.shape}")
    return embeddings_array
