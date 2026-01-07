"""Smoke test: run end-to-end pipeline on sample docs."""

import pytest

from finance_rag_eval.constants import EVAL_GOLD_SET, SAMPLE_DOCS_DIR
from finance_rag_eval.eval.runner import evaluate_config
from finance_rag_eval.rag.chunking import chunk_documents
from finance_rag_eval.rag.embeddings import generate_embeddings
from finance_rag_eval.rag.index import FAISSIndex
from finance_rag_eval.rag.ingestion import load_documents_from_dir


def test_load_sample_docs():
    """Test loading sample documents."""
    documents = load_documents_from_dir(SAMPLE_DOCS_DIR)
    assert len(documents) > 0
    assert all("id" in doc for doc in documents)
    assert all("text" in doc for doc in documents)


def test_build_index_smoke():
    """Smoke test: build index from sample docs."""
    documents = load_documents_from_dir(SAMPLE_DOCS_DIR)
    assert len(documents) > 0

    chunks = chunk_documents(documents, chunk_size=256, chunk_overlap=50)
    assert len(chunks) > 0

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)
    assert embeddings.shape[0] == len(chunks)

    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)
    assert index.index.ntotal == len(chunks)


def test_eval_smoke():
    """Smoke test: run evaluation on sample config."""
    if not EVAL_GOLD_SET.exists():
        pytest.skip("Gold set not found")

    config = {
        "chunk_size": 256,
        "retriever": "cosine",
        "top_k": 3,
        "rerank": False,
    }

    results = evaluate_config(config, SAMPLE_DOCS_DIR, EVAL_GOLD_SET)

    assert "error" not in results
    assert "avg_context_recall" in results
    assert "avg_faithfulness" in results
    assert "p50_latency" in results
    assert results["num_questions"] > 0
