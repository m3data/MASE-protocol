"""
Embedding service for MASE semantic analysis.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Uses sentence-transformers with all-mpnet-base-v2 for embedding generation.

Model selection rationale (from Semantic Climate diagnostic):
- Highest velocity variance (most sensitive to local semantic shifts)
- Best local-vs-distant gap (preserves structural relationships)
- Valid alpha (DFA) values
- Mid-range similarity (~0.22) - doesn't collapse distinctions

Retrieval-optimized models (Nomic, mxbai) collapse local semantic motion,
which undermines curvature and DFA measurements.
"""

import numpy as np
from typing import List, Optional, Union


class EmbeddingService:
    """
    Generate embeddings using sentence-transformers.

    Provides a consistent interface for embedding text for semantic analysis.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: sentence-transformers model identifier.
                Default "all-mpnet-base-v2" (768 dimensions).
                Alternative: "all-MiniLM-L6-v2" (384 dimensions, faster).
        """
        self.model_name = model_name
        self._model = None
        self._dimensions: Optional[int] = None

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        self._ensure_loaded()
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array (shape: dimensions,)
        """
        self._ensure_loaded()
        return self._model.encode(text)

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            show_progress: Show progress bar for large batches

        Returns:
            Array of embedding vectors (shape: len(texts), dimensions)
        """
        self._ensure_loaded()
        return self._model.encode(texts, show_progress_bar=show_progress)

    def semantic_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine distance between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine distance (1 - cosine_similarity), range [0, 2]
        """
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return 1.0 - similarity

    def semantic_velocity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute semantic velocity (consecutive cosine distances).

        Args:
            embeddings: Array of embeddings (shape: n_turns, dimensions)

        Returns:
            Array of velocities (shape: n_turns - 1,)
        """
        if len(embeddings) < 2:
            return np.array([])

        velocities = []
        for i in range(1, len(embeddings)):
            dist = self.semantic_distance(embeddings[i-1], embeddings[i])
            velocities.append(dist)

        return np.array(velocities)


# Singleton for convenience
_default_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = "all-mpnet-base-v2") -> EmbeddingService:
    """
    Get or create a default embedding service.

    Uses singleton pattern to avoid reloading model.
    """
    global _default_service
    if _default_service is None or _default_service.model_name != model_name:
        _default_service = EmbeddingService(model_name)
    return _default_service


def embed(text: Union[str, List[str]]) -> np.ndarray:
    """
    Convenience function to embed text.

    Args:
        text: Single string or list of strings

    Returns:
        Embedding(s) as numpy array
    """
    service = get_embedding_service()
    if isinstance(text, str):
        return service.embed(text)
    return service.embed_batch(text)


# Test if run directly
if __name__ == "__main__":
    print("Embedding Service Test")
    print("=" * 50)

    service = EmbeddingService()

    print(f"\nModel: {service.model_name}")
    print(f"Dimensions: {service.dimensions}")

    # Test single embedding
    text1 = "The quick brown fox jumps over the lazy dog."
    emb1 = service.embed(text1)
    print(f"\nSingle embedding shape: {emb1.shape}")

    # Test batch embedding
    texts = [
        "Hello world",
        "Goodbye world",
        "The universe is vast and mysterious"
    ]
    embeddings = service.embed_batch(texts)
    print(f"Batch embedding shape: {embeddings.shape}")

    # Test semantic distance
    dist = service.semantic_distance(embeddings[0], embeddings[1])
    print(f"\nDistance 'Hello' <-> 'Goodbye': {dist:.4f}")

    dist = service.semantic_distance(embeddings[0], embeddings[2])
    print(f"Distance 'Hello' <-> 'Universe': {dist:.4f}")

    # Test velocity
    velocity = service.semantic_velocity(embeddings)
    print(f"\nSemantic velocity: {velocity}")
