"""
RAG Pipeline Module
===================

Combines all components into a working RAG system:
Question → Retrieve relevant docs → Generate answer with context
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Prompt template for RAG
PROMPT_TEMPLATE = """You are a helpful assistant for TechCorp.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """A complete RAG pipeline that retrieves context and generates answers."""

    def __init__(self, vector_store, embedding_func):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: A SimpleVectorStore instance with documents
            embedding_func: Function to generate embeddings (get_embedding)
        """
        self.vector_store = vector_store
        self.get_embedding = embedding_func
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def query(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using retrieved context.

        Steps:
        1. Embed the question
        2. Search for relevant documents
        3. Build prompt with context
        4. Generate answer

        Args:
            question: The user's question
            top_k: Number of documents to retrieve

        Returns:
            The generated answer
        """
        # 1. Embed the question
        question_embedding = self.get_embedding(question, task_type="retrieval_query")

        # 2. Search for relevant documents
        relevant_docs = self.vector_store.search(question_embedding, top_k=top_k)

        # 3. Build prompt with context
        prompt = self.build_prompt(relevant_docs, question)

        # 4. Generate answer
        response = self.model.generate_content(prompt)

        return response.text

    def build_prompt(self, contexts: list[str], question: str) -> str:
        """
        Create a prompt with context and question.

        Args:
            contexts: List of relevant document chunks
            question: The user's question

        Returns:
            The formatted prompt
        """
        # Join all context documents with separators
        context_text = "\n\n---\n\n".join(contexts)

        # Fill in the template
        return PROMPT_TEMPLATE.format(context=context_text, question=question)
