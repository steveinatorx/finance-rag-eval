"""Evaluation metrics: context recall, faithfulness, latency."""

import time
from typing import Dict, List

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def context_recall_proxy(
    retrieved_chunks: List[dict],
    gold_answer: str,
) -> float:
    """
    Proxy for context recall: check if retrieved chunks contain answer spans.

    Simple heuristic: check if key phrases from gold answer appear in retrieved chunks.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        gold_answer: Gold standard answer

    Returns:
        Recall score between 0 and 1
    """
    if not retrieved_chunks or not gold_answer:
        return 0.0

    # Extract key phrases from gold answer (words longer than 4 chars)
    gold_lower = gold_answer.lower()
    key_phrases = [
        phrase.strip() for phrase in gold_lower.split() if len(phrase.strip()) > 4
    ]

    if not key_phrases:
        return 0.0

    # Check how many key phrases appear in retrieved chunks
    retrieved_text = " ".join(
        [chunk["chunk"]["text"].lower() for chunk in retrieved_chunks]
    )

    matches = sum(1 for phrase in key_phrases if phrase in retrieved_text)
    recall = matches / len(key_phrases) if key_phrases else 0.0

    return min(recall, 1.0)


def faithfulness_proxy(
    answer: str,
    retrieved_chunks: List[dict],
) -> float:
    """
    Proxy for faithfulness: check if answer sentences are supported by context.

    Simple heuristic: check if answer sentences contain words/phrases from retrieved chunks.

    Args:
        answer: Generated answer
        retrieved_chunks: List of retrieved chunk dictionaries

    Returns:
        Faithfulness score between 0 and 1
    """
    if not answer or not retrieved_chunks:
        return 0.0

    # Extract sentences from answer
    import re

    answer_sentences = re.split(r"[.!?]+", answer)
    answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

    if not answer_sentences:
        return 0.0

    # Build context from retrieved chunks
    context_text = " ".join(
        [chunk["chunk"]["text"].lower() for chunk in retrieved_chunks]
    )

    # Check each sentence for support
    supported_sentences = 0
    for sentence in answer_sentences:
        sentence_lower = sentence.lower()
        # Extract meaningful words (length > 3)
        words = [w for w in sentence_lower.split() if len(w) > 3]
        if words:
            # Check if at least some words appear in context
            matches = sum(1 for w in words if w in context_text)
            if matches >= len(words) * 0.3:  # At least 30% of words match
                supported_sentences += 1

    faithfulness = (
        supported_sentences / len(answer_sentences) if answer_sentences else 0.0
    )
    return faithfulness


def measure_latency(func, *args, **kwargs) -> tuple:
    """
    Measure latency of a function call.

    Args:
        func: Function to measure
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Tuple of (result, latency_seconds)
    """
    start = time.time()
    result = func(*args, **kwargs)
    latency = time.time() - start
    return result, latency


def multi_document_coverage(
    retrieved_chunks: List[dict],
    required_documents: List[str],
) -> float:
    """
    Measure how well multi-document queries retrieve from required documents.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        required_documents: List of document IDs that should be retrieved

    Returns:
        Coverage score between 0 and 1 (percentage of required docs retrieved)
    """
    if not required_documents:
        return 1.0  # No requirements = perfect coverage

    # Extract document sources from retrieved chunks
    retrieved_sources = set()
    for chunk in retrieved_chunks:
        source = chunk.get("chunk", {}).get("metadata", {}).get("source", "")
        # Extract doc ID from source path (e.g., "doc1.txt" from full path)
        if source:
            import os

            doc_id = os.path.basename(source)
            retrieved_sources.add(doc_id)

    # Check coverage
    required_set = set(required_documents)
    covered = len(required_set & retrieved_sources)
    coverage = covered / len(required_set) if required_set else 1.0

    return coverage


def compute_metrics(
    query: str,
    gold_answer: str,
    retrieved_chunks: List[dict],
    answer: str,
    retrieval_latency: float,
    generation_latency: float,
    question_type: str = None,
    required_documents: List[str] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        query: Query string
        gold_answer: Gold standard answer
        retrieved_chunks: Retrieved chunks
        answer: Generated answer
        retrieval_latency: Retrieval latency in seconds
        generation_latency: Generation latency in seconds
        question_type: Optional type ('multi_document', 'temporal', etc.)
        required_documents: Optional list of required document IDs for multi-doc queries

    Returns:
        Dictionary of metric names to values
    """
    recall = context_recall_proxy(retrieved_chunks, gold_answer)
    faithfulness = faithfulness_proxy(answer, retrieved_chunks)
    total_latency = retrieval_latency + generation_latency

    metrics = {
        "context_recall": recall,
        "faithfulness": faithfulness,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "total_latency": total_latency,
    }

    # Add multi-document coverage if applicable
    if question_type == "multi_document" and required_documents:
        metrics["multi_doc_coverage"] = multi_document_coverage(
            retrieved_chunks, required_documents
        )

    return metrics
