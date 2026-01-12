"""Chunking quality evaluation: intrinsic metrics that don't require gold sets."""

import re
from typing import Dict, List

import numpy as np

from finance_rag_eval.logging import get_logger
from finance_rag_eval.rag.embeddings import generate_embeddings

logger = get_logger(__name__)


def measure_chunk_size_distribution(chunks: List[dict]) -> Dict[str, float]:
    """
    Measure chunk size distribution.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with size metrics
    """
    sizes = [len(chunk.get("text", "")) for chunk in chunks]

    if not sizes:
        return {}

    return {
        "chunk_count": len(chunks),
        "avg_chunk_size": float(np.mean(sizes)),
        "chunk_size_std": float(np.std(sizes)),
        "min_chunk_size": float(np.min(sizes)),
        "max_chunk_size": float(np.max(sizes)),
        "median_chunk_size": float(np.median(sizes)),
        "p25_chunk_size": float(np.percentile(sizes, 25)),
        "p75_chunk_size": float(np.percentile(sizes, 75)),
    }


def measure_semantic_coherence(chunks: List[dict]) -> Dict[str, float]:
    """
    Measure semantic coherence within chunks.

    Higher values indicate chunks contain semantically related content.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with coherence metrics
    """
    if not chunks:
        return {}

    try:
        # Split chunks into sentences
        chunk_sentences = []
        for chunk in chunks:
            text = chunk.get("text", "")
            # Simple sentence splitting
            sentences = re.split(r"[.!?]+\s+", text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if sentences:
                chunk_sentences.append(sentences)

        if not chunk_sentences:
            return {"avg_intra_chunk_coherence": 0.0, "coherent_chunks": 0}

        # Generate embeddings for all sentences
        all_sentences = [s for sentences in chunk_sentences for s in sentences]
        if len(all_sentences) < 2:
            return {"avg_intra_chunk_coherence": 0.0, "coherent_chunks": 0}

        embeddings = generate_embeddings(all_sentences)

        # Calculate intra-chunk similarity
        intra_chunk_similarities = []
        sentence_idx = 0

        for sentences in chunk_sentences:
            if len(sentences) < 2:
                sentence_idx += len(sentences)
                continue

            chunk_embeddings = embeddings[sentence_idx : sentence_idx + len(sentences)]
            sentence_idx += len(sentences)

            # Calculate pairwise similarities within chunk
            similarities = []
            for i in range(len(chunk_embeddings)):
                for j in range(i + 1, len(chunk_embeddings)):
                    sim = np.dot(chunk_embeddings[i], chunk_embeddings[j]) / (
                        np.linalg.norm(chunk_embeddings[i])
                        * np.linalg.norm(chunk_embeddings[j])
                        + 1e-8
                    )
                    similarities.append(sim)

            if similarities:
                intra_chunk_similarities.append(np.mean(similarities))

        if not intra_chunk_similarities:
            return {"avg_intra_chunk_coherence": 0.0, "coherent_chunks": 0}

        return {
            "avg_intra_chunk_coherence": float(np.mean(intra_chunk_similarities)),
            "coherent_chunks": len(
                [s for s in intra_chunk_similarities if s > 0.5]
            ),  # Threshold
        }

    except Exception as e:
        logger.warning(f"Error measuring semantic coherence: {e}")
        return {"avg_intra_chunk_coherence": 0.0, "coherent_chunks": 0}


def measure_boundary_quality(chunks: List[dict]) -> Dict[str, float]:
    """
    Measure chunk boundary quality.

    Checks if boundaries align with natural breaks (sentence/paragraph boundaries).

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with boundary quality metrics
    """
    if not chunks:
        return {}

    # Check if chunk boundaries align with sentence/paragraph breaks
    boundary_aligned = 0
    total_boundaries = len(chunks) - 1

    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].get("text", "")
        next_chunk = chunks[i + 1].get("text", "")

        # Check if current chunk ends with sentence boundary
        current_ends_well = (
            current_chunk.rstrip().endswith(".")
            or current_chunk.rstrip().endswith("?")
            or current_chunk.rstrip().endswith("!")
            or current_chunk.rstrip().endswith("\n\n")
        )

        # Check if next chunk starts with capital letter (new sentence)
        next_starts_well = (
            len(next_chunk.strip()) > 0 and next_chunk.strip()[0].isupper()
        ) or next_chunk.strip().startswith("\n")

        if current_ends_well or next_starts_well:
            boundary_aligned += 1

    return {
        "boundary_alignment_ratio": (
            boundary_aligned / total_boundaries if total_boundaries > 0 else 0.0
        ),
        "aligned_boundaries": boundary_aligned,
        "total_boundaries": total_boundaries,
    }


def measure_structure_preservation(chunks: List[dict]) -> Dict[str, float]:
    """
    Measure if document structure (headers, tables) is preserved.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with structure preservation metrics
    """
    if not chunks:
        return {}

    headers_preserved = 0
    tables_preserved = 0
    chunks_with_structure = 0

    for chunk in chunks:
        text = chunk.get("text", "").lower()
        has_header = "table:" in text or any(
            keyword in text
            for keyword in [
                "consolidated statements",
                "fiscal year",
                "item",
                "part",
            ]
        )
        has_table = "|" in text and any(
            keyword in text
            for keyword in ["2023", "2024", "2025", "total", "net sales"]
        )

        if has_header:
            headers_preserved += 1
        if has_table:
            tables_preserved += 1
        if has_header or has_table:
            chunks_with_structure += 1

    return {
        "chunks_with_headers": headers_preserved,
        "chunks_with_tables": tables_preserved,
        "chunks_with_structure": chunks_with_structure,
        "structure_preservation_ratio": (
            chunks_with_structure / len(chunks) if chunks else 0.0
        ),
    }


def measure_document_coverage(
    chunks: List[dict], documents: List[dict]
) -> Dict[str, float]:
    """
    Measure how well chunks cover the source documents.

    Args:
        chunks: List of chunk dictionaries
        documents: List of source documents

    Returns:
        Dictionary with coverage metrics
    """
    if not chunks or not documents:
        return {}

    # Calculate total document length
    total_doc_length = sum(len(doc.get("text", "")) for doc in documents)

    # Calculate total chunk length (accounting for overlap)
    chunk_texts = set()  # Use set to deduplicate if needed
    for chunk in chunks:
        chunk_texts.add(chunk.get("text", ""))

    total_chunk_length = sum(len(text) for text in chunk_texts)

    # Calculate coverage
    coverage_ratio = (
        total_chunk_length / total_doc_length if total_doc_length > 0 else 0.0
    )

    return {
        "total_doc_length": total_doc_length,
        "total_chunk_length": total_chunk_length,
        "coverage_ratio": coverage_ratio,
        "chunks_per_doc": len(chunks) / len(documents) if documents else 0.0,
    }


def evaluate_chunking_quality(
    chunks: List[dict], documents: List[dict], strategy_name: str = None
) -> Dict[str, any]:
    """
    Comprehensive chunking quality evaluation.

    Intrinsic metrics that don't require gold sets.

    Args:
        chunks: List of chunk dictionaries
        documents: List of source documents
        strategy_name: Optional name of chunking strategy

    Returns:
        Dictionary with all chunking quality metrics
    """
    logger.info(f"Evaluating chunking quality for {len(chunks)} chunks")

    metrics = {
        "strategy": strategy_name,
        "size_metrics": measure_chunk_size_distribution(chunks),
        "coherence_metrics": measure_semantic_coherence(chunks),
        "boundary_metrics": measure_boundary_quality(chunks),
        "structure_metrics": measure_structure_preservation(chunks),
        "coverage_metrics": measure_document_coverage(chunks, documents),
    }

    logger.info(
        f"Chunking quality: "
        f"avg_size={metrics['size_metrics'].get('avg_chunk_size', 0):.0f}, "
        f"coherence={metrics['coherence_metrics'].get('avg_intra_chunk_coherence', 0):.3f}, "
        f"boundary_quality={metrics['boundary_metrics'].get('boundary_alignment_ratio', 0):.3f}, "
        f"structure_preservation={metrics['structure_metrics'].get('structure_preservation_ratio', 0):.3f}"
    )

    return metrics
