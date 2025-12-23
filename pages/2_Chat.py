"""
Chat Interface Page
===================

This page provides a chat interface to ask questions using the RAG pipeline.
"""

import streamlit as st
import time
from src.rag.data_loader import load_documents, split_into_chunks
from src.rag.embeddings import get_embedding
from src.rag.vector_store import SimpleVectorStore
from src.rag.pipeline import RAGPipeline


@st.cache_resource
def initialize_rag():
    """Initialize the RAG pipeline (cached to avoid reloading)."""
    # Load and process documents
    documents = load_documents("data/documents")

    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc, chunk_size=500, overlap=50)
        all_chunks.extend(chunks)

    # Create vector store and add embeddings
    store = SimpleVectorStore()
    progress_bar = st.progress(0, text="Indexing documents...")

    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk)
        store.add(chunk, embedding)
        progress_bar.progress(
            (i + 1) / len(all_chunks), text=f"Indexing... {i + 1}/{len(all_chunks)}"
        )

    progress_bar.empty()

    # Create and return pipeline
    return RAGPipeline(store, get_embedding), all_chunks


# Page header
st.title("💬 Chat with TechCorp Assistant")
st.markdown("*Ask questions about TechCorp products, policies, and more*")

st.divider()

# Initialize pipeline
with st.spinner("🔧 Loading RAG pipeline (first time only)..."):
    rag, all_chunks = initialize_rag()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("💡 Example Questions")
example_questions = [
    "What products does TechCorp offer?",
    "What is the refund policy?",
    "How can I contact support?",
    "Is my data secure?",
    "How much does DataFlow Pro cost?",
]

for q in example_questions:
    if st.sidebar.button(q, use_container_width=True, key=f"btn_{q}"):
        st.session_state.messages.append({"role": "user", "content": q})
        st.rerun()

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f"**📚 Knowledge Base**")
st.sidebar.markdown(f"- {len(all_chunks)} chunks indexed")
st.sidebar.markdown(f"- 3 documents loaded")

# Check if we need to generate a response for the last message
needs_response = (
    len(st.session_state.messages) > 0
    and st.session_state.messages[-1]["role"] == "user"
    and not any(
        m["role"] == "assistant"
        for m in st.session_state.messages[-2:]
        if len(st.session_state.messages) > 1
    )
)

# If last message is from user without a response, check if it's truly pending
if len(st.session_state.messages) > 0:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "user":
        # Check if there's no assistant response after this user message
        needs_response = True
        for i, m in enumerate(st.session_state.messages):
            if (
                i > 0
                and st.session_state.messages[i - 1]["role"] == "user"
                and m["role"] == "assistant"
            ):
                if st.session_state.messages[i - 1]["content"] == last_msg["content"]:
                    needs_response = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "time" in message:
            st.caption(f"⏱️ {message['time']:.2f}s")

# Generate response if needed (from sidebar button click)
if needs_response and len(st.session_state.messages) > 0:
    last_user_msg = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            start_time = time.time()
            response = rag.query(last_user_msg)
            elapsed = time.time() - start_time
        st.markdown(response)
        st.caption(f"⏱️ {elapsed:.2f}s")
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "time": elapsed}
    )
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about TechCorp..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            start_time = time.time()
            response = rag.query(prompt)
            elapsed = time.time() - start_time

        st.markdown(response)
        st.caption(f"⏱️ {elapsed:.2f}s")

    # Add assistant message to history
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "time": elapsed}
    )
