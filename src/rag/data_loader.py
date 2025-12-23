"""
Data Loader Module
==================

This module handles loading and processing text documents for RAG:
1. Reading text files from the data/documents/ directory
2. Splitting documents into smaller chunks (for better retrieval)

Why chunking matters:
- LLMs have limited context windows
- Smaller chunks = more precise retrieval
- Typical chunk sizes: 200-1000 characters with some overlap
"""

from pathlib import Path


def load_documents(directory: str) -> list[str]:
    """
    Load all .txt files from a directory.

    Args:
        directory: Path to the directory containing .txt files

    Returns:
        A list of document contents (one string per file)
    """
    documents = []
    dir_path = Path(directory)

    for file_path in dir_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)

    return documents


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into smaller chunks with optional overlap.

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        A list of text chunks
    """
    # First, try to split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk_size, save current chunk
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        # If a single paragraph is larger than chunk_size, split it by size
        while len(current_chunk) > chunk_size:
            # Find a good break point (space) near chunk_size
            break_point = current_chunk.rfind(" ", 0, chunk_size)
            if break_point == -1:
                break_point = chunk_size

            chunks.append(current_chunk[:break_point].strip())
            # Keep overlap for continuity
            start = max(0, break_point - overlap)
            current_chunk = current_chunk[start:].strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
