"""FAISS index builder and management with optional BM25 support for hybrid retrieval."""

import pickle
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)

# Lazy loading for BM25
_bm25_available = None


def _check_bm25_available():
    """Check if rank-bm25 is available."""
    global _bm25_available
    if _bm25_available is None:
        try:
            import rank_bm25  # noqa: F401

            _bm25_available = True
        except ImportError:
            _bm25_available = False
    return _bm25_available


class FAISSIndex:
    """FAISS index wrapper for vector similarity search with optional BM25 support."""

    def __init__(self, dimension: int, enable_bm25: bool = False):
        """
        Initialize FAISS index.

        Args:
            dimension: Dimension of embeddings
            enable_bm25: If True, also build BM25 index for hybrid retrieval
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[dict] = []
        self.enable_bm25 = enable_bm25 and _check_bm25_available()
        self.bm25_index = None
        self.bm25_tokenized_corpus = None

    def add(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        """
        Add embeddings and associated chunks to index.

        Args:
            embeddings: numpy array of embeddings (n_chunks, dimension)
            chunks: List of chunk dictionaries
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        # Normalize embeddings for cosine similarity (L2 normalization)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

        # Build BM25 index if enabled
        if self.enable_bm25:
            self._build_bm25_index(chunks)

        logger.info(
            f"Added {len(chunks)} chunks to index (total: {self.index.ntotal}, "
            f"BM25: {self.enable_bm25})"
        )

    def _build_bm25_index(self, chunks: List[dict]) -> None:
        """Build BM25 index from chunks."""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize chunks for BM25
            tokenized_corpus = []
            for chunk in chunks:
                text = chunk.get("text", "")
                # Clean HTML: remove tags, decode entities, normalize whitespace
                text_clean = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
                text_clean = re.sub(
                    r"&#[0-9]+;", " ", text_clean
                )  # Remove HTML entities
                text_clean = re.sub(r"\s+", " ", text_clean)  # Normalize whitespace

                # Extract tokens: words and numbers (including numbers with commas)
                # This regex matches: word characters OR numbers (with optional commas/spaces)
                tokens = re.findall(
                    r"\b\w+\b|\d{1,3}(?:[,\s]\d{3})*", text_clean.lower()
                )
                # Also extract individual number parts (383, 285) for better matching
                number_parts = re.findall(r"\d+", text_clean)
                tokens.extend(number_parts)

                tokenized_corpus.append(tokens)

            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_tokenized_corpus = tokenized_corpus
            logger.debug("Built BM25 index")
        except ImportError:
            logger.warning(
                "rank-bm25 not available. Install with: pip install rank-bm25"
            )
            self.enable_bm25 = False

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector (1, dimension)
            k: Number of results to return

        Returns:
            List of chunk dictionaries with 'chunk', 'score', 'metadata'
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        distances, indices = self.index.search(
            query_embedding, min(k, self.index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (1 - normalized distance)
                score = 1.0 / (1.0 + dist)
                results.append(
                    {
                        "chunk": chunk,
                        "score": float(score),
                        "metadata": chunk.get("metadata", {}),
                    }
                )

        return results

    def bm25_search(self, query: str, k: int = 5) -> List[dict]:
        """
        Search using BM25 (keyword-based retrieval).

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of chunk dictionaries with 'chunk', 'score', 'metadata'
        """
        if not self.enable_bm25 or self.bm25_index is None:
            return []

        # Tokenize query: extract words and numbers
        query_clean = query.lower()
        query_tokens = re.findall(r"\b\w+\b|\d{1,3}(?:[,\s]\d{3})*", query_clean)
        # Also extract individual number parts
        number_parts = re.findall(r"\d+", query_clean)
        query_tokens.extend(number_parts)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx < len(self.chunks) and scores[idx] > 0:
                chunk = self.chunks[idx]
                # Normalize BM25 score (BM25 scores can be > 1, normalize to 0-1)
                max_score = max(scores) if len(scores) > 0 else 1.0
                normalized_score = (
                    min(scores[idx] / max_score, 1.0) if max_score > 0 else 0.0
                )
                results.append(
                    {
                        "chunk": chunk,
                        "score": float(normalized_score),
                        "metadata": chunk.get("metadata", {}),
                    }
                )

        return results

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Save index and chunks to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}")

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "FAISSIndex":
        """Load index and chunks from disk."""
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # Infer dimension from index
        dimension = index.d
        instance = cls(dimension)
        instance.index = index
        instance.chunks = chunks

        logger.info(f"Loaded index from {index_path} with {len(chunks)} chunks")
        return instance
