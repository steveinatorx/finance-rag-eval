"""Dagster assets for FAISS index building."""

from typing import List

import numpy as np
from dagster import AssetExecutionContext, ResourceParam, asset

from finance_rag_eval.dagster_app.resources import PathsResource
from finance_rag_eval.rag.index import FAISSIndex


@asset(key_prefix=["rag"])
def faiss_index(
    context: AssetExecutionContext,
    chunks: List[dict],
    embeddings: np.ndarray,
    paths: ResourceParam[PathsResource],
) -> FAISSIndex:
    """
    Build FAISS index from chunks and embeddings.

    Args:
        chunks: List of chunk dictionaries
        embeddings: Embedding vectors
        paths: Paths resource

    Returns:
        FAISSIndex instance
    """
    context.log.info(f"Building FAISS index with {len(chunks)} chunks")

    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)

    # Optionally save index
    outputs_dir = paths.get_outputs_dir()
    index_path = outputs_dir / "faiss_index.index"
    chunks_path = outputs_dir / "chunks.pkl"

    index.save(index_path, chunks_path)
    context.log.info(f"Saved index to {index_path}")

    return index
