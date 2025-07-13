# Claude RAG + Pinecone Integration Guide

Perfect! Since your documents are already in Pinecone, you can connect Claude RAG directly to your existing vector database. This guide will help you set up the integration.

## 🚀 Quick Setup

### 1. Install Pinecone Dependencies

```bash
# Update your installation to include Pinecone support
pip install pinecone-client>=3.0.0

# Or reinstall everything
python3 install.py
```

### 2. Add Pinecone API Key

Edit your `.env` file to include your Pinecone API key:

```bash
# Your existing keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Add Pinecone API key
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Get your Pinecone API key:**
1. Go to [app.pinecone.io](https://app.pinecone.io/)
2. Log in to your account
3. Navigate to "API Keys" 
4. Copy your API key

### 3. Run the Pinecone Example

```bash
python3 example_pinecone.py
```

The script will ask for:
- Your Pinecone index name
- Namespace (optional)

Then you can immediately start querying your existing documents!

## 📋 How It Works

### Architecture with Pinecone

```
Your Documents (Pinecone) ←→ Claude RAG ←→ Claude API
     ↓                            ↓
   Existing Embeddings      Local Query Embeddings
```

### What Happens When You Query

1. **Your question** → Local embedding model (sentence-transformers)
2. **Query embedding** → Pinecone similarity search  
3. **Relevant documents** → Retrieved from your Pinecone index
4. **Context + Question** → Sent to Claude API
5. **Intelligent answer** → Generated based on your documents

## 🔧 Configuration Options

### Basic Usage

```python
from claude_rag_pinecone import ClaudeRAGPinecone

# Connect to your existing Pinecone index
rag = ClaudeRAGPinecone(
    index_name="your-index-name",    # Your existing index
    namespace="your-namespace",      # Optional namespace
    embedding_model="all-MiniLM-L6-v2"  # Local embedding model
)

# Query your existing documents
answer = rag.query_existing_documents("What is machine learning?")
print(answer)
```

### Advanced Configuration

```python
# With custom settings
rag = ClaudeRAGPinecone(
    anthropic_api_key="your-key",
    pinecone_api_key="your-key", 
    index_name="my-docs",
    namespace="technical-docs",
    embedding_model="all-MiniLM-L6-v2"  # or try "all-mpnet-base-v2"
)

# Query with filtering
answer = rag.query_existing_documents(
    "Tell me about Python",
    k=10,  # Retrieve top 10 most similar docs
    max_context_length=6000,  # More context for Claude
    filter={"category": {"$eq": "programming"}}  # Filter by metadata
)
```

## 📊 Working with Your Data

### Querying Existing Documents

This is the main use case - your documents are already in Pinecone:

```python
# Simple query
answer = rag.query_existing_documents("Your question here")

# Advanced query with options
answer = rag.query_existing_documents(
    question="Your question",
    k=5,                              # Number of similar docs to retrieve
    max_context_length=4000,          # Token limit for context
    filter={"source": {"$eq": "wiki"}} # Metadata filtering
)
```

### Adding New Documents (Optional)

If you want to add more documents to your existing index:

```python
new_docs = ["Document 1 content", "Document 2 content"]
sources = ["source1", "source2"]
doc_ids = ["doc1", "doc2"]

rag.add_documents(new_docs, sources, doc_ids)
```

### Filtering by Metadata

If your Pinecone vectors have metadata, you can filter results:

```python
# Only search documents from specific sources
filter_by_source = {"source": {"$eq": "research_papers"}}

# Only search recent documents  
filter_by_date = {"date": {"$gte": "2023-01-01"}}

# Multiple conditions
complex_filter = {
    "$and": [
        {"category": {"$eq": "technical"}},
        {"language": {"$eq": "english"}}
    ]
}

answer = rag.query_existing_documents(
    "Your question",
    filter=filter_by_source
)
```

## 🔄 Embedding Compatibility

### Important: Embedding Models

Your Pinecone vectors were created with a specific embedding model. For best results:

**If your Pinecone vectors use:**
- **OpenAI embeddings (text-embedding-ada-002):** Use dimension 1536
- **Sentence Transformers:** Check the model name and dimension
- **Other models:** Match the original embedding model

### Checking Your Index

```python
# Get information about your index
stats = rag.get_stats()
print(stats)

# Output example:
# {
#   'total_vectors': 1250,
#   'dimension': 384,
#   'embedding_model': 'all-MiniLM-L6-v2',
#   'pinecone_index': 'my-docs',
#   'namespace': 'default'
# }
```

### Embedding Model Options

```python
# Smaller, faster model (384 dimensions)
rag = ClaudeRAGPinecone(embedding_model="all-MiniLM-L6-v2")

# Larger, more accurate model (768 dimensions)  
rag = ClaudeRAGPinecone(embedding_model="all-mpnet-base-v2")

# If you need to match OpenAI embeddings, consider:
# rag = ClaudeRAGPinecone(embedding_model="text-embedding-ada-002")  # Requires OpenAI API
```

## 🎯 Best Practices

### 1. Namespace Organization

Use namespaces to organize different document types:

```python
# Different RAG instances for different document types
tech_rag = ClaudeRAGPinecone(index_name="docs", namespace="technical")
business_rag = ClaudeRAGPinecone(index_name="docs", namespace="business")

tech_answer = tech_rag.query_existing_documents("How does API work?")
business_answer = business_rag.query_existing_documents("What's our revenue?")
```

### 2. Metadata Filtering

Structure your metadata for effective filtering:

```python
# Good metadata structure
metadata = {
    "source": "research_paper",
    "category": "machine_learning", 
    "date": "2023-12-01",
    "author": "Smith et al.",
    "confidence": 0.95
}

# Query specific document types
ml_papers = rag.query_existing_documents(
    "What are recent advances in ML?",
    filter={"category": {"$eq": "machine_learning"}}
)
```

### 3. Optimizing Retrieval

```python
# For factual questions: fewer, more relevant docs
answer = rag.query_existing_documents("What is the capital of France?", k=3)

# For complex analysis: more docs for broader context  
answer = rag.query_existing_documents("Compare ML frameworks", k=10)

# For long documents: increase context length
answer = rag.query_existing_documents(
    "Summarize the research findings", 
    max_context_length=8000
)
```

## 🛠️ Troubleshooting

### Common Issues

**Error: "Index not found"**
- Check your index name in Pinecone console
- Verify your API key has access to the index

**Error: "Dimension mismatch"**
- Your embedding model dimensions don't match Pinecone index
- Check your index dimension in Pinecone console
- Use the same embedding model that created your vectors

**Poor Results**
- Try different embedding models
- Increase `k` (number of retrieved documents)
- Adjust `max_context_length`
- Check if your documents have good metadata for filtering

### Debugging

```python
# Check connection and stats
try:
    stats = rag.get_stats()
    print(f"Connected! Total vectors: {stats['total_vectors']}")
except Exception as e:
    print(f"Connection failed: {e}")

# Test similarity search directly  
query_embedding = rag.embedding_generator.generate_embedding("test query")
results = rag.vector_store.similarity_search(query_embedding, k=5)
print(f"Found {len(results)} similar documents")
```

## 📈 Performance Tips

1. **Batch Queries:** If querying multiple questions, reuse the RAG instance
2. **Namespace Strategy:** Use namespaces to reduce search space
3. **Metadata Filtering:** Pre-filter documents for better relevance
4. **Context Length:** Balance between detail and speed
5. **Embedding Caching:** The local embedding model caches automatically

## 🔄 Migration from Other Systems

### From OpenAI Embeddings

If your Pinecone vectors use OpenAI embeddings:

```python
# Option 1: Use OpenAI embeddings (requires API key)
from claude_rag import ClaudeRAG  # Original version

# Option 2: Re-embed with local model (one-time process)
# This requires re-processing your documents
```

### From Other Vector DBs

The same principles apply to other vector databases. You can adapt the `PineconeVectorStore` class for:
- Weaviate
- Chroma  
- FAISS
- Qdrant

## 🎉 You're Ready!

With your documents already in Pinecone, you have a powerful setup:

✅ **No embedding costs** - Local embeddings for queries  
✅ **Existing data** - Your documents are ready to query  
✅ **Scalable** - Pinecone handles large document collections  
✅ **Fast retrieval** - Optimized vector search  
✅ **Intelligent responses** - Claude generates answers from your data  

Start with:
```bash
python3 example_pinecone.py
```

And begin asking questions about your documents! 