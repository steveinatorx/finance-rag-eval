"""Optional cross-encoder reranking (gated by availability)."""

from typing import List, Optional

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)

# Lazy loading
_cross_encoder_model = None


def _get_cross_encoder_model():
    """Lazy load cross-encoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker")
            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            logger.warning("sentence-transformers not available for reranking")
            return None
    return _cross_encoder_model


def rerank(
    query: str,
    retrieved_chunks: List[dict],
    top_k: Optional[int] = None,
) -> List[dict]:
    """
    Rerank retrieved chunks using a cross-encoder.

    Args:
        query: Query string
        retrieved_chunks: List of retrieved chunk dictionaries
        top_k: Optional limit on number of results

    Returns:
        Reranked list of chunks
    """
    model = _get_cross_encoder_model()

    if model is None:
        logger.debug("Reranking skipped (model not available)")
        return retrieved_chunks

    if not retrieved_chunks:
        return []

    # Prepare pairs for cross-encoder
    pairs = [[query, chunk["chunk"]["text"]] for chunk in retrieved_chunks]

    # Get relevance scores
    scores = model.predict(pairs)

    # Update scores and sort
    for chunk, score in zip(retrieved_chunks, scores):
        chunk["score"] = float(score)

    # Sort by score (descending)
    reranked = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)

    if top_k:
        reranked = reranked[:top_k]

    logger.debug(f"Reranked {len(retrieved_chunks)} chunks")
    return reranked
