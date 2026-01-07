"""LangChain chunking integration (optional)."""

from typing import List

from finance_rag_eval.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def langchain_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "recursive",
) -> List[str]:
    """
    Chunk text using LangChain's text splitters.

    Requires: pip install langchain

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: 'recursive', 'semantic', or 'markdown'

    Returns:
        List of text chunks
    """
    try:
        from langchain.text_splitter import (
            MarkdownTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # Try to import semantic splitter (may not be available in all versions)
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            HAS_SEMANTIC = True
        except ImportError:
            HAS_SEMANTIC = False
            logger.debug("LangChain semantic chunker not available")

    except ImportError:
        logger.warning(
            "LangChain not installed. Install with: pip install langchain langchain-experimental"
        )
        # Fallback to our custom chunking
        from finance_rag_eval.rag.chunking import fixed_size_chunk

        return fixed_size_chunk(text, chunk_size, chunk_overlap)

    if strategy == "recursive":
        # Recursive character text splitter (most common)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

    elif strategy == "markdown":
        # Markdown-aware splitter
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(text)

    elif strategy == "semantic" and HAS_SEMANTIC:
        # Semantic chunking (requires embeddings)
        try:
            from finance_rag_eval.rag.embeddings import _get_sentence_model

            embedding_model = _get_sentence_model()

            text_splitter = SemanticChunker.from_tiktoken_encoder(
                embedding=embedding_model,
                chunk_size=chunk_size,
            )
            chunks = text_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to recursive")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = text_splitter.split_text(text)
    else:
        # Fallback to recursive
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(text)

    return chunks


def chunk_documents_langchain(
    documents: List[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "recursive",
) -> List[dict]:
    """
    Chunk documents using LangChain.

    Args:
        documents: List of document dictionaries
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        strategy: 'recursive', 'semantic', or 'markdown'

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", "")
        text_chunks = langchain_chunk(text, chunk_size, chunk_overlap, strategy)

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": f"{doc['id']}_chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    **doc.get("metadata", {}),
                    "doc_id": doc["id"],
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "strategy": f"langchain_{strategy}",
                },
            }
            chunks.append(chunk)

    logger.info(
        f"Chunked {len(documents)} documents into {len(chunks)} chunks using LangChain {strategy}"
    )
    return chunks
