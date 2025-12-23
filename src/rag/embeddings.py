"""
Embeddings Module
=================

This module generates text embeddings using Gemini's embedding model.
Embeddings are numerical representations of text (vectors of 768 numbers).
Similar texts have similar embeddings, allowing semantic search.
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Generate an embedding vector for a text.

    Args:
        text: The text to embed
        task_type:
            - "retrieval_document" for documents you're storing
            - "retrieval_query" for user questions

    Returns:
        A list of 768 floats representing the text
    """
    result = genai.embed_content(
        model="models/embedding-001", content=text, task_type=task_type
    )
    return result["embedding"]


def get_embeddings_batch(
    texts: list[str], task_type: str = "retrieval_document"
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        task_type: Same as get_embedding

    Returns:
        A list of embeddings (one per text)
    """
    embeddings = []
    for text in texts:
        embedding = get_embedding(text, task_type)
        embeddings.append(embedding)
    return embeddings
