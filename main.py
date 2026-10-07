#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Application
Uses llama-cpp-python for LLM inference and ChromaDB for vector storage
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Import packages (will use .venv if available)
try:
    import chromadb
    from llama_cpp import Llama
    print("✓ Packages imported successfully")
except ImportError as e:
    print(f"Error importing packages: {e}")
    print("Make sure you've run: pip install llama-cpp-python chromadb")
    exit(1)


def load_embeddings(model_name="llama-3.2-3b-instruct") -> Llama:
    """Initialize the LLM model"""
    print(f"Loading model: {model_name}")

    # Configure for CPU-only usage (no GPU acceleration)
    llm = Llama(
        model_name=model_name,
        n_ctx=4096,  # Context window size
        verbose=False,
        n_threads=4,
        n_gpu_split_raw=0,  # CPU only
    )

    print("Model loaded successfully")
    return llm


def create_chroma_collection(name="rag_db", persist_dir="data"):
    """Create or load ChromaDB collection"""
    db = chromadb.EphemeralClient()
    collection = db.create_collection(name=name)
    return collection


def add_documents(collection, documents, metadata=None):
    """Add documents to the vector database"""
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Prepare embeddings (would normally generate them here with llama_cpp)
    embeddings = [[1.0]*384] * len(documents)  # Placeholder embeddings

    # Add documents to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=[metadata] if metadata else []
    )

    print(f"Added {len(documents)} documents")


def retrieve_relevant_docs(collection, query_text, k=3):
    """Retrieve relevant documents for a query"""
    # For real usage, you'd pass your own embeddings here
    # This is a placeholder showing the API call
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results


def chat_with_rag(llm, collection, user_query):
    """Combine retrieved context with LLM for answer generation"""
    # Retrieve relevant documents
    results = retrieve_relevant_docs(collection, user_query)

    if not results or len(results['documents']) == 0:
        response = llm(
            prompt=f"Question: {user_query}\n\nNo relevant context was found. Please ask a different question.",
            max_tokens=512,
            stop=["\n"]
        )
        return response["output"]

    # Build context-aware prompt
    context_docs = "\n\n".join(results['documents'][0])
    prompt = f"""Context: {context_docs}

User Question: {user_query}

Based on the context above, provide a concise and accurate answer. If the context doesn't contain enough information to answer the question, state that clearly."""

    response = llm(
        prompt=prompt,
        max_tokens=512,
        stop=["\n"]
    )

    return response["output"]


def ingest_file(filepath, collection):
    """Ingest a text file into the RAG system"""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    # Read and split content
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into chunks (simple word-by-word for demo)
    words = content.split()
    chunks = [f"Chunk {i+1}: {words[i]}".split(': ') if i % 5 == 0 else
              f"Chunk {i+1}: {' '.join(words[i:i+5])}"
              for i in range(0, len(words), 5)]

    # Add to database
    add_documents(collection, chunks, {"source": str(path)})
    print(f"Ingested file: {filepath}")


if __name__ == "__main__":
    # Initialize components
    print("=" * 60)
    print("RAG Application - Retrieval-Augmented Generation")
    print("=" * 60)

    llm = load_embeddings(model_name="llama-3.2-3b-instruct-q4_0")
    collection = create_chroma_collection(name="rag_db", persist_dir="data")

    # Example: Ingest some sample data
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is widely used in machine learning due to its readability.",
        "Deep learning uses neural networks with multiple layers.",
    ]

    print("\n--- Demo Mode ---")
    ingest_file("/dev/stdin", collection)

    # Ask a question
    print("\nEnter a question to query the RAG system (or 'exit' to quit):")

    while True:
        query = input("> ").strip()

        if query.lower() in ('exit', 'quit'):
            break

        answer = chat_with_rag(llm, collection, query)
        print(f"\nAnswer: {answer}\n")
