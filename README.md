# Claude RAG System

A complete Retrieval-Augmented Generation (RAG) system using Claude API directly, without external frameworks like LangChain or LlamaIndex.

## Features

- **Document Processing**: Intelligent text chunking with overlaps for better context preservation
- **Vector Embeddings**: Uses OpenAI's text-embedding-ada-002 for high-quality embeddings
- **Similarity Search**: Cosine similarity-based retrieval of relevant document chunks
- **Claude Integration**: Direct integration with Anthropic's Claude API for response generation
- **Modular Design**: Clean, extensible architecture for easy customization

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

Edit `.env` with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Getting API Key:**
- **Anthropic API Key**: Sign up at [console.anthropic.com](https://console.anthropic.com)

**Note:** This version uses local embeddings (sentence-transformers), so you only need the Anthropic API key!

### 3. Run the Example

**Local embeddings version (recommended):**
```bash
python3 example_local_rag.py
```

**Or the original version (requires OpenAI API key):**
```bash
python3 example_usage.py
```

## Usage

### Basic Usage

**Using local embeddings (no OpenAI API key needed):**

```python
from claude_rag_no_openai import ClaudeRAGLocal

# Initialize the RAG system (only needs Anthropic API key)
rag = ClaudeRAGLocal()

# Add documents
documents = [
    "Your first document content here...",
    "Your second document content here...",
    # ... more documents
]

sources = ["doc1", "doc2"]  # Optional source identifiers

# Add documents to the system
rag.add_documents(documents, sources)

# Query the system
answer = rag.query("What is artificial intelligence?")
print(answer)
```

**Using OpenAI embeddings (requires both API keys):**

```python
from claude_rag import ClaudeRAG

# Initialize the RAG system
rag = ClaudeRAG()
# ... rest is the same
```

### Advanced Usage

```python
# Custom initialization with specific API keys
rag = ClaudeRAG(
    anthropic_api_key="your_anthropic_key",
    openai_api_key="your_openai_key"
)

# Customize chunking parameters
from claude_rag import DocumentProcessor
processor = DocumentProcessor(
    chunk_size=1500,  # tokens per chunk
    chunk_overlap=300  # overlap between chunks
)

# Query with custom parameters
answer = rag.query(
    "Your question here",
    max_context_length=6000,  # max tokens in context
    k=3  # number of chunks to retrieve
)
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
- Simple in-memory vector store using NumPy
- Supports cosine similarity search
- Easily extensible to external vector databases

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

### `ClaudeRAG`
Main orchestrator class that coordinates all components.

### `DocumentProcessor`
Handles text chunking and preprocessing with configurable parameters.

### `EmbeddingGenerator`
Manages embedding generation using OpenAI's API with batch processing support.

### `VectorStore`
Simple in-memory storage for document embeddings with similarity search.

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

- **In-Memory Storage**: Not suitable for very large document collections
- **No Persistence**: Embeddings are lost when the program ends
- **Simple Retrieval**: Uses basic cosine similarity; no advanced ranking

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