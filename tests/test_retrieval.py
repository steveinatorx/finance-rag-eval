"""Tests for retrieval functionality."""

import numpy as np

from finance_rag_eval.rag.index import FAISSIndex
from finance_rag_eval.rag.retrieval import cosine_retrieval, mmr_retrieval


def test_faiss_index():
    """Test FAISS index creation and search."""
    dimension = 128
    index = FAISSIndex(dimension)

    # Create dummy embeddings and chunks
    embeddings = np.random.randn(10, dimension).astype("float32")
    chunks = [
        {"id": f"chunk_{i}", "text": f"Text {i}", "metadata": {}} for i in range(10)
    ]

    index.add(embeddings, chunks)
    assert index.index.ntotal == 10

    # Search
    query_embedding = np.random.randn(1, dimension).astype("float32")
    results = index.search(query_embedding, k=3)

    assert len(results) <= 3
    assert all("chunk" in r for r in results)
    assert all("score" in r for r in results)


def test_cosine_retrieval():
    """Test cosine similarity retrieval."""
    dimension = 128
    index = FAISSIndex(dimension)

    embeddings = np.random.randn(5, dimension).astype("float32")
    chunks = [
        {"id": f"chunk_{i}", "text": f"Text {i}", "metadata": {}} for i in range(5)
    ]

    index.add(embeddings, chunks)

    query_embedding = np.random.randn(1, dimension).astype("float32")
    results = cosine_retrieval(query_embedding, index, k=3)

    assert len(results) <= 3
    # Results should be sorted by score (descending)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_mmr_retrieval():
    """Test MMR retrieval."""
    dimension = 128
    index = FAISSIndex(dimension)

    embeddings = np.random.randn(10, dimension).astype("float32")
    chunks = [
        {"id": f"chunk_{i}", "text": f"Text {i}", "metadata": {}} for i in range(10)
    ]

    index.add(embeddings, chunks)

    query_embedding = np.random.randn(1, dimension).astype("float32")
    results = mmr_retrieval(query_embedding, index, k=5, diversity=0.5)

    assert len(results) <= 5
    assert all("chunk" in r for r in results)
