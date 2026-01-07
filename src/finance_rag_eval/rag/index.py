"""FAISS index builder and management."""

import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


class FAISSIndex:
    """FAISS index wrapper for vector similarity search."""

    def __init__(self, dimension: int):
        """
        Initialize FAISS index.

        Args:
            dimension: Dimension of embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[dict] = []

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
        logger.info(f"Added {len(chunks)} chunks to index (total: {self.index.ntotal})")

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
