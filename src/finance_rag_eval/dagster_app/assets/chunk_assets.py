"""Dagster assets for document chunking."""

from typing import List

from dagster import AssetExecutionContext, asset

from finance_rag_eval.constants import DEFAULT_CHUNK_SIZE
from finance_rag_eval.rag.chunking import chunk_documents


@asset(key_prefix=["rag"])
def chunks(context: AssetExecutionContext, docs_clean: List[dict]) -> List[dict]:
    """
    Chunk documents into smaller pieces.

    Args:
        docs_clean: Cleaned documents

    Returns:
        List of chunk dictionaries
    """
    context.log.info(f"Chunking {len(docs_clean)} documents")

    chunks_list = chunk_documents(
        docs_clean,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=50,
        strategy="fixed",
    )

    context.log.info(f"Created {len(chunks_list)} chunks")
    return chunks_list
