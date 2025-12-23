"""
Vector Store Module
===================

A simple vector store for similarity search using numpy.
Stores text chunks and their embeddings, and finds similar documents.
"""

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between -1 and 1 (1 = identical)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SimpleVectorStore:
    """A simple in-memory vector store for similarity search."""

    def __init__(self):
        self.texts = []  # Store the original text chunks
        self.embeddings = []  # Store the embeddings

    def add(self, text: str, embedding: list[float]):
        """
        Add a text and its embedding to the store.

        Args:
            text: The original text chunk
            embedding: The embedding vector for this text
        """
        self.texts.append(text)
        self.embeddings.append(embedding)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        """
        Find the top_k most similar texts to the query.

        Args:
            query_embedding: The embedding of the search query
            top_k: Number of results to return

        Returns:
            List of the most similar text chunks
        """
        if not self.embeddings:
            return []

        # Calculate similarity with all stored embeddings
        similarities = []
        for embedding in self.embeddings:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append(sim)

        # Get indices of top_k most similar (highest scores)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return the corresponding texts
        return [self.texts[i] for i in top_indices]
