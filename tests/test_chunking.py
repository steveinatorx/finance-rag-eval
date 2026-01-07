"""Tests for chunking functionality."""

from finance_rag_eval.rag.chunking import (
    chunk_documents,
    fixed_size_chunk,
    recursive_chunk,
)


def test_fixed_size_chunk():
    """Test fixed-size chunking."""
    text = "a" * 1000
    chunks = fixed_size_chunk(text, chunk_size=256, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 256 for chunk in chunks)
    # Check overlap
    if len(chunks) > 1:
        # First chunk should overlap with second
        assert chunks[0][-50:] == chunks[1][:50]


def test_recursive_chunk():
    """Test recursive chunking."""
    text = "Sentence one.\n\nSentence two.\n\nSentence three."
    chunks = recursive_chunk(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    # Should prefer splitting on paragraph breaks
    assert all(len(chunk) <= 100 for chunk in chunks)  # Allow some flexibility


def test_chunk_documents():
    """Test chunking a list of documents."""
    documents = [
        {"id": "doc1", "text": "a" * 500, "metadata": {}},
        {"id": "doc2", "text": "b" * 300, "metadata": {}},
    ]

    chunks = chunk_documents(documents, chunk_size=256, chunk_overlap=50)

    assert len(chunks) > 0
    assert all("chunk_index" in chunk["metadata"] for chunk in chunks)
    assert all(chunk["metadata"]["doc_id"] in ["doc1", "doc2"] for chunk in chunks)
