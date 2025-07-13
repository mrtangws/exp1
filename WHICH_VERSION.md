# Which Claude RAG Version Should You Use?

Since you only have an Anthropic API key (no OpenAI), you should use the **local embeddings version**. Here's what you need to know:

## 🎯 Recommended: Local Embeddings Version

**File:** `claude_rag_no_openai.py`  
**Example:** `example_local_rag.py`

### ✅ Advantages
- **Only needs Anthropic API key** - no OpenAI account required
- **No usage costs for embeddings** - runs locally on your machine
- **Privacy friendly** - your documents never leave your computer for embedding
- **Fast after setup** - no API calls for embeddings once model is downloaded
- **High quality** - uses state-of-the-art sentence-transformers models

### ⚠️ Considerations
- **Initial download** - ~90MB model download on first run
- **CPU usage** - embedding generation uses your computer's CPU
- **Slightly slower startup** - model loading takes a few seconds

### 📦 What It Uses
- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Embedding Dimension:** 384
- **Model Size:** ~90MB
- **Language Model:** Claude 3 Sonnet (via Anthropic API)

## 🔄 Alternative: OpenAI Embeddings Version

**File:** `claude_rag.py`  
**Example:** `example_usage.py`

### ✅ Advantages
- **No local model download** - uses OpenAI's cloud embeddings
- **Slightly higher quality embeddings** - OpenAI's text-embedding-ada-002 is very good
- **No local CPU usage** - embeddings computed in the cloud

### ❌ Disadvantages
- **Requires OpenAI API key** - additional account and billing setup
- **Usage costs** - small cost per embedding (~$0.0001 per 1K tokens)
- **Privacy considerations** - documents sent to OpenAI for embedding
- **API dependency** - requires internet connection for embeddings

## 🚀 Getting Started (Local Version)

1. **Install dependencies:**
   ```bash
   python3 install.py
   ```

2. **Set up your Anthropic API key:**
   ```bash
   cp env_example.txt .env
   # Edit .env and add: ANTHROPIC_API_KEY=your_key_here
   ```

3. **Run the example:**
   ```bash
   python3 example_local_rag.py
   ```

## 📊 Performance Comparison

| Feature | Local Embeddings | OpenAI Embeddings |
|---------|-----------------|-------------------|
| Setup complexity | Easy | Medium (2 API keys) |
| Ongoing costs | Free | ~$0.0001 per 1K tokens |
| Privacy | High | Medium |
| Embedding quality | High | Very High |
| Startup time | 3-5 seconds | Instant |
| Processing speed | Fast | Very Fast |
| Offline capability | Yes (after setup) | No |

## 🔧 Code Examples

### Local Version
```python
from claude_rag_no_openai import ClaudeRAGLocal

rag = ClaudeRAGLocal()  # Only needs ANTHROPIC_API_KEY
rag.add_documents(["Your documents..."])
answer = rag.query("Your question?")
```

### OpenAI Version  
```python
from claude_rag import ClaudeRAG

rag = ClaudeRAG()  # Needs ANTHROPIC_API_KEY + OPENAI_API_KEY
rag.add_documents(["Your documents..."])
answer = rag.query("Your question?")
```

## 🎯 Recommendation

**Use the local embeddings version** (`claude_rag_no_openai.py`) since:
- You only have an Anthropic API key
- It's simpler to set up
- No ongoing costs
- Better privacy
- Still provides excellent results

The quality difference between the two embedding approaches is minimal for most use cases, and the local version is more convenient for your situation. 