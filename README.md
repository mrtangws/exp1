# Claude RAG System

A complete Retrieval-Augmented Generation (RAG) system using Claude API with Pinecone vector database integration.

## Features

- **Document Processing**: Intelligent text chunking with overlaps for better context preservation
- **Vector Embeddings**: Support for both local embeddings (sentence-transformers) and OpenAI embeddings
- **Pinecone Integration**: Production-ready vector database for scalable document storage
- **Similarity Search**: Cosine similarity-based retrieval of relevant document chunks
- **Claude Integration**: Direct integration with Anthropic's Claude API for response generation
- **Flexible Architecture**: Support for multiple embedding models and vector stores

## Quick Start

**⚠️ Having installation issues with Python 3.13? See [INSTALL.md](INSTALL.md) for detailed troubleshooting.**

### Automatic Installation (Recommended)
```bash
python install.py
```

### Manual Installation
```bash
# 1. Update build tools first
pip install --upgrade pip setuptools wheel

# 2. Install dependencies  
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API key:

```bash
cp env_example.txt .env
```

Edit `.env` with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for OpenAI embeddings
```

**Getting API Keys:**
- **Anthropic API Key**: Sign up at [console.anthropic.com](https://console.anthropic.com)
- **Pinecone API Key**: Sign up at [pinecone.io](https://pinecone.io)
- **OpenAI API Key**: Sign up at [platform.openai.com](https://platform.openai.com) (optional)

### 3. Run the Examples

**Pinecone RAG (recommended for production):**
```bash
python example_pinecone.py
```

**Local RAG (for testing):**
```bash
python example_local_rag.py
```

**Ingest documents into Pinecone:**
```bash
python ingest_documents.py
```

## Usage

### Basic Usage with Pinecone

```python
from claude_rag_pinecone import ClaudeRAGPinecone

# Initialize with Pinecone
rag = ClaudeRAGPinecone(
    index_name="your-index",
    namespace="your-namespace",
    embedding_model="sentence-transformers/all-roberta-large-v1",  # 1024 dims
    claude_model="claude-3-5-sonnet-20241022"
)

# Add documents to Pinecone
documents = ["Your document content..."]
rag.add_documents(documents, sources=["doc1"], doc_id_prefixes=["doc_1"])

# Query existing documents
answer = rag.query_existing_documents(
    "Your question?",
    k=5,  # Number of documents to retrieve
    system_instructions="You are a helpful assistant."
)
print(answer)
```

### Ingesting Documents

```python
# Ingest from folder
from ingest_documents import ingest_text_files

# With chunking
ingest_text_files("./documents", "your-index", "your-namespace")

# Without chunking
ingest_text_files("./documents", "your-index", "your-namespace", no_chunking=True)
```

### Advanced Usage

```python
# Custom initialization with OpenAI embeddings
rag = ClaudeRAGPinecone(
    index_name="your-index",
    namespace="your-namespace",
    embedding_model="text-embedding-ada-002",
    claude_model="claude-3-5-sonnet-20241022",
    use_openai_embeddings=True  # Requires 1536-dim index
)

# Query with filtering
answer = rag.query_existing_documents(
    "Your question",
    k=10,
    filter={"source": {"$eq": "specific_document.pdf"}},
    system_instructions="You are a medical expert."
)

# Search all documents
total_docs = rag.get_stats().get('total_vectors', 5)
answer = rag.query_existing_documents("Your question", k=total_docs)
```

## How It Works

### 1. Document Processing
- Documents are split into overlapping chunks (default: 1000 tokens, 200 overlap)
- Chunks preserve sentence boundaries for better coherence
- Each chunk includes metadata (source, token count)

### 2. Embedding Generation
- Uses OpenAI's `text-embedding-ada-002` model
- Generates 1536-dimensional vectors for each chunk
- Supports batch processing for efficiency

### 3. Vector Storage
- Production-ready Pinecone vector database
- Supports cosine similarity search with metadata filtering
- Scalable storage for large document collections
- Namespace support for document organization

### 4. Retrieval & Generation
- Query is embedded using the same embedding model
- Cosine similarity finds most relevant chunks
- Context is built respecting token limits
- Claude generates response based on retrieved context

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Document        │───▶│   Embeddings    │
│                 │    │  Processor       │    │   Generator     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Similarity      │◀───│  Vector Store   │
│                 │    │  Search          │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  Retrieved      │
         │              │  Context        │
         │              └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Claude API                                   │
│                 (Response Generation)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### `ClaudeRAGPinecone`
Main orchestrator class that coordinates all components with Pinecone integration.

### `DocumentProcessor`
Handles text chunking and preprocessing with configurable parameters.

### `LocalEmbeddingGenerator` / `OpenAIEmbeddingGenerator`
Manages embedding generation using either local sentence-transformers or OpenAI API.

### `PineconeVectorStore`
Production-ready vector storage using Pinecone with metadata filtering and namespaces.

### `Document`
Data class representing a document chunk with metadata and embeddings.

## Customization

### Using Different Embedding Models

```python
# Modify EmbeddingGenerator class
class CustomEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.model = "text-embedding-3-small"  # Different model
```

### Adding External Vector Databases

```python
# Example: Pinecone integration
import pinecone

class PineconeVectorStore(VectorStore):
    def __init__(self, index_name):
        pinecone.init(api_key="your-key", environment="your-env")
        self.index = pinecone.Index(index_name)
    
    def add_documents(self, documents):
        # Implementation for Pinecone
        pass
```

### Custom Chunking Strategies

```python
class CustomDocumentProcessor(DocumentProcessor):
    def chunk_text(self, text, source=""):
        # Your custom chunking logic
        pass
```

## Performance Considerations

- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Memory Usage**: In-memory vector store; consider external databases for large datasets
- **Token Limits**: Context length is managed to stay within Claude's limits
- **Caching**: Consider caching embeddings for frequently accessed documents

## Limitations

- **Dimension Matching**: Embedding model dimensions must match Pinecone index dimensions
- **API Dependencies**: Requires active Pinecone and Anthropic API keys
- **Simple Retrieval**: Uses basic cosine similarity; no advanced ranking algorithms

## Future Enhancements

- [ ] Persistent vector storage
- [ ] Multiple embedding model support
- [ ] Advanced retrieval algorithms (MMR, etc.)
- [ ] Document metadata filtering
- [ ] Conversation memory
- [ ] Hybrid search (keyword + semantic)

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - see LICENSE file for details. 