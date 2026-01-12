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

    Improved heuristic: check if key phrases from gold answer appear in retrieved chunks,
    with better handling of numbers, formatting, and HTML entities.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        gold_answer: Gold standard answer

    Returns:
        Recall score between 0 and 1
    """
    if not retrieved_chunks or not gold_answer:
        return 0.0

    import re

    # Normalize text: remove HTML entities, normalize whitespace
    def normalize_text(text: str) -> str:
        # Remove HTML entities
        text = re.sub(r"&#\d+;", " ", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    gold_normalized = normalize_text(gold_answer)
    retrieved_text = " ".join(
        [normalize_text(chunk["chunk"]["text"]) for chunk in retrieved_chunks]
    )

    # Extract key phrases: words longer than 3 chars
    key_phrases = []
    for word in gold_normalized.split():
        word_clean = re.sub(r"[^\w]", "", word)  # Remove punctuation
        if len(word_clean) > 3 and not word_clean.isdigit():
            key_phrases.append(word_clean)

    # Extract numbers (normalize: remove commas, spaces, dollar signs)
    gold_numbers = re.findall(r"[\d,]+", gold_answer)
    gold_numbers_clean = [
        num.replace(",", "").replace(" ", "")
        for num in gold_numbers
        if len(num.replace(",", "")) >= 3
    ]

    # Normalize retrieved text for number matching
    retrieved_numbers_normalized = re.sub(r"[^\d]", "", retrieved_text)

    # Count matches
    phrase_matches = sum(1 for phrase in key_phrases if phrase in retrieved_text)

    # Check if numbers appear (at least one number must match)
    number_matches = 0
    if gold_numbers_clean:
        for num in gold_numbers_clean:
            if num in retrieved_numbers_normalized:
                number_matches = 1  # Count as 1 if any number matches
                break

    # Calculate recall: weighted combination
    # Phrases are more important, but numbers are critical
    total_phrases = len(key_phrases) if key_phrases else 1
    phrase_score = phrase_matches / total_phrases

    # If there are numbers, they must be present for full score
    if gold_numbers_clean:
        number_score = number_matches  # 0 or 1
        # Weighted: 70% phrases, 30% numbers (numbers are critical)
        recall = 0.7 * phrase_score + 0.3 * number_score
    else:
        # No numbers, just use phrase score
        recall = phrase_score

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
    query: str,  # pylint: disable=unused-argument
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
