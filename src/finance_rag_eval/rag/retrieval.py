"""Retrieval strategies: cosine similarity and MMR (Maximal Marginal Relevance)."""

from typing import List

import numpy as np

from finance_rag_eval.constants import DEFAULT_MMR_DIVERSITY, DEFAULT_TOP_K
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def cosine_retrieval(
    query_embedding: np.ndarray,
    index,
    k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Retrieve top-k chunks using cosine similarity.

    Args:
        query_embedding: Query embedding vector
        index: FAISSIndex instance
        k: Number of results

    Returns:
        List of retrieved chunks with scores
    """
    results = index.search(query_embedding, k)
    return results


def mmr_retrieval(
    query_embedding: np.ndarray,
    index,
    k: int = DEFAULT_TOP_K,
    diversity: float = DEFAULT_MMR_DIVERSITY,
) -> List[dict]:
    """
    Retrieve chunks using Maximal Marginal Relevance (MMR).

    MMR balances relevance and diversity by selecting chunks that are:
    - Relevant to the query
    - Different from already selected chunks

    Args:
        query_embedding: Query embedding vector
        index: FAISSIndex instance
        k: Number of results
        diversity: Diversity parameter (0-1, higher = more diverse)

    Returns:
        List of retrieved chunks with scores
    """
    # Get more candidates than needed for MMR selection
    candidates = index.search(query_embedding, k * 3)

    if not candidates:
        return []

    selected = []
    remaining = candidates.copy()

    while len(selected) < k and remaining:
        if not selected:
            # First item: highest relevance
            best = remaining[0]
            selected.append(best)
            remaining.remove(best)
        else:
            # Compute MMR score for each remaining candidate
            best_score = -float("inf")
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance to query
                relevance = candidate["score"]

                # Max similarity to already selected chunks
                max_sim = 0.0
                if selected:
                    # Get embedding for candidate (simplified: use score as proxy)
                    # In a full implementation, would compute actual cosine similarity
                    # between candidate embedding and selected embeddings
                    for sel in selected:
                        # Approximate similarity using score difference
                        sim = 1.0 - abs(candidate["score"] - sel["score"])
                        max_sim = max(max_sim, sim)

                # MMR score: relevance - diversity * max_similarity
                mmr_score = relevance - diversity * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

    return selected


def retrieve(
    query_embedding: np.ndarray,
    index,
    k: int = DEFAULT_TOP_K,
    strategy: str = "cosine",
    diversity: float = DEFAULT_MMR_DIVERSITY,
) -> List[dict]:
    """
    Retrieve chunks using specified strategy.

    Args:
        query_embedding: Query embedding vector
        index: FAISSIndex instance
        k: Number of results
        strategy: 'cosine' or 'mmr'
        diversity: MMR diversity parameter (only used for MMR)

    Returns:
        List of retrieved chunks
    """
    if strategy == "mmr":
        return mmr_retrieval(query_embedding, index, k, diversity)
    else:
        return cosine_retrieval(query_embedding, index, k)
