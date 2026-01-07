"""Dagster assets for document ingestion."""

from typing import List

from dagster import AssetExecutionContext, ResourceParam, asset

from finance_rag_eval.dagster_app.resources import PathsResource
from finance_rag_eval.rag.ingestion import load_documents_from_dir


@asset(key_prefix=["rag"])
def docs_raw(context: AssetExecutionContext, paths: ResourceParam[PathsResource]) -> List[dict]:
    """
    Load raw documents from sample docs directory.

    Returns:
        List of document dictionaries
    """
    docs_dir = paths.get_sample_docs_dir()
    context.log.info(f"Loading documents from {docs_dir}")

    documents = load_documents_from_dir(docs_dir)
    context.log.info(f"Loaded {len(documents)} raw documents")

    return documents


@asset(key_prefix=["rag"])
def docs_clean(context: AssetExecutionContext, docs_raw: List[dict]) -> List[dict]:
    """
    Clean documents (currently just passes through, but could add cleaning logic).

    Args:
        docs_raw: Raw documents

    Returns:
        Cleaned documents
    """
    context.log.info(f"Cleaning {len(docs_raw)} documents")
    # For now, just return as-is (cleaning happens in ingestion)
    return docs_raw
