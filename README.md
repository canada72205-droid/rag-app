# RAG (Retrieval-Augmented Generation) Application

A simple Retrieval-Augmented Generation system using:
- **llama-cpp-python** - For local LLM inference
- **chromadb** - For vector database and document storage

## Installation

Make sure you're in the correct directory with a virtual environment activated:

```bash
cd "/Users/tylerrop/Desktop/MU/LatestTrends/Rag"
source .venv/bin/activate
```

Install dependencies if needed:

```bash
pip install llama-cpp-python chromadb
```

## Usage

### Basic Chat

```python
from main import chat_with_rag

# Initialize the RAG system
rag = chat_with_rag()

# Ask a question
answer = rag("What is mentioned in the documents about topic X?")
print(answer)
```

### Adding Documents

```python
from main import add_documents

# Add a list of text chunks
add_documents([
    "Document 1 content here...",
    "Document 2 content here..."
])
```

### Ingesting Files

```python
from main import ingest_file

# Ingest a text file (auto-detected extension: .txt, .md)
ingest_file("/path/to/your/document.txt")
```

## Available Functions

- `load_embeddings()` - Initialize the llama-cpp LLM model
- `create_chroma_collection()` - Set up ChromaDB vector storage
- `add_documents()` - Add text chunks to vector database
- `retrieve_relevant_docs()` - Search for relevant documents
- `chat_with_rag()` - Combined retrieval + LLM interface
- `ingest_file()` - Ingest text files into the system

## Dependencies

- llama-cpp-python (LLM inference)
- chromadb (vector database)
- PyYAML (optional, for config loading)
