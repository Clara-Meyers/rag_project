"""
RAG Tutorial Project - Main Entry Point
=======================================

Interactive Q&A application using RAG (Retrieval-Augmented Generation).
Ask questions about TechCorp and get answers based on the company documents!
"""

from src.rag.data_loader import load_documents, split_into_chunks
from src.rag.embeddings import get_embedding
from src.rag.vector_store import SimpleVectorStore
from src.rag.pipeline import RAGPipeline


def main():
    """Main function - runs the interactive RAG Q&A application."""

    print("=" * 60)
    print("🚀 TechCorp RAG Assistant")
    print("=" * 60)
    print()

    # 1. Load all documents
    print("📚 Loading documents...")
    documents = load_documents("data/documents")
    print(f"   ✅ Loaded {len(documents)} documents")

    # 2. Split into chunks
    print("✂️  Splitting into chunks...")
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc, chunk_size=500, overlap=50)
        all_chunks.extend(chunks)
    print(f"   ✅ Created {len(all_chunks)} chunks")

    # 3. Generate embeddings and store them
    print("🧠 Generating embeddings (this may take a moment)...")
    store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk)
        store.add(chunk, embedding)
        # Progress indicator
        if (i + 1) % 5 == 0 or i == len(all_chunks) - 1:
            print(f"   📊 Progress: {i + 1}/{len(all_chunks)} chunks indexed")
    print(f"   ✅ All chunks indexed!")

    # 4. Create RAG pipeline
    print("🔧 Creating RAG pipeline...")
    rag = RAGPipeline(store, get_embedding)
    print("   ✅ Pipeline ready!")

    print()
    print("=" * 60)
    print("💬 Ask me anything about TechCorp!")
    print("   Type 'quit' to exit")
    print("=" * 60)

    # 5. Interactive question loop
    while True:
        print()
        question = input("❓ Your question: ")

        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not question.strip():
            print("   Please enter a question.")
            continue

        print("\n🔍 Searching for relevant information...")
        answer = rag.query(question)
        print(f"\n💬 Answer:\n{answer}")


if __name__ == "__main__":
    main()
