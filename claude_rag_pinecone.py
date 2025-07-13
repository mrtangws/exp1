import os
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import tiktoken
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

@dataclass
class Document:
    """Represents a document chunk with its metadata"""
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_id: str = ""

class DocumentProcessor:
    """Handles document chunking and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def chunk_text(self, text: str, source: str = "", doc_id_prefix: str = "") -> List[Document]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                doc_id = f"{doc_id_prefix}_chunk_{chunk_idx}" if doc_id_prefix else f"chunk_{chunk_idx}"
                chunks.append(Document(
                    content=current_chunk.strip(),
                    metadata={"tokens": current_tokens, "source": source, "chunk_index": chunk_idx},
                    source=source,
                    doc_id=doc_id
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.tokenizer.encode(current_chunk))
                chunk_idx += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            doc_id = f"{doc_id_prefix}_chunk_{chunk_idx}" if doc_id_prefix else f"chunk_{chunk_idx}"
            chunks.append(Document(
                content=current_chunk.strip(),
                metadata={"tokens": current_tokens, "source": source, "chunk_index": chunk_idx},
                source=source,
                doc_id=doc_id
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.tokenizer.decode(overlap_tokens)

class LocalEmbeddingGenerator:
    """Handles text embedding generation using local Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        print(f"Loading embedding model: {model_name}")
        print("This may take a moment on first run...")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Embedding model loaded: {model_name}")
        print(f"   Embedding dimension: {self.embedding_dimension}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension or 384)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [self.generate_embedding(text) for text in texts]

class PineconeVectorStore:
    """Pinecone vector store integration"""
    
    def __init__(self, 
                 api_key: str = None, 
                 index_name: str = "claude-rag",
                 environment: str = "gcp-starter",
                 embedding_dimension: int = 384):
        
        if not HAS_PINECONE:
            raise ImportError(
                "pinecone-client is required. Install with: pip install pinecone-client"
            )
        
        # Initialize Pinecone
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY in environment.")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        # Get or create index
        self._setup_index()
        
    def _setup_index(self):
        """Setup Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Change as needed
                    )
                )
                print(f"✅ Created index: {self.index_name}")
            else:
                print(f"✅ Using existing index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            raise
    
    def add_documents(self, documents: List[Document], namespace: str = ""):
        """Add documents to Pinecone"""
        vectors = []
        
        for doc in documents:
            if doc.embedding is not None:
                # Prepare metadata for Pinecone
                metadata = {
                    "content": doc.content,
                    "source": doc.source,
                    **doc.metadata  # Include any additional metadata
                }
                
                vectors.append({
                    "id": doc.doc_id,
                    "values": doc.embedding.tolist(),
                    "metadata": metadata
                })
        
        if vectors:
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            print(f"✅ Added {len(vectors)} documents to Pinecone")
    
    def similarity_search(self, 
                         query_embedding: np.ndarray, 
                         k: int = 5, 
                         namespace: str = "",
                         filter: Dict = None) -> List[Tuple[Document, float]]:
        """Search for similar documents in Pinecone"""
        try:
            # Query Pinecone
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True,
                namespace=namespace,
                filter=filter
            )
            
            # Convert results to Document objects
            results = []
            for match in response.matches:
                doc = Document(
                    content=match.metadata.get("content", ""),
                    source=match.metadata.get("source", ""),
                    metadata=match.metadata,
                    doc_id=match.id
                )
                results.append((doc, match.score))
            
            return results
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
    
    def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}

class ClaudeRAGPinecone:
    """Claude RAG system with Pinecone vector database integration"""
    
    def __init__(self, 
                 anthropic_api_key: str = None,
                 pinecone_api_key: str = None,
                 index_name: str = "claude-rag",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 namespace: str = ""):
        
        # Setup Anthropic client
        anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("Anthropic API key is required")
        
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Setup components
        self.processor = DocumentProcessor()
        self.embedding_generator = LocalEmbeddingGenerator(embedding_model)
        
        # Setup Pinecone
        self.vector_store = PineconeVectorStore(
            api_key=pinecone_api_key,
            index_name=index_name,
            embedding_dimension=self.embedding_generator.embedding_dimension
        )
        
        self.namespace = namespace
        
        print(f"✅ Claude RAG with Pinecone initialized")
        print(f"   Index: {index_name}")
        print(f"   Namespace: {namespace or 'default'}")
    
    def add_documents(self, texts: List[str], sources: List[str] = None, doc_id_prefixes: List[str] = None) -> None:
        """Add new documents to Pinecone"""
        if sources is None:
            sources = [f"document_{i}" for i in range(len(texts))]
        
        if doc_id_prefixes is None:
            doc_id_prefixes = [f"doc_{i}" for i in range(len(texts))]
        
        if sources is None:
            sources = [f"document_{i}" for i in range(len(texts))]
        
        print(f"Processing {len(texts)} documents...")
        
        # Chunk documents
        all_chunks = []
        for text, source, doc_prefix in zip(texts, sources, doc_id_prefixes):
            chunks = self.processor.chunk_text(text, source, doc_prefix)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to Pinecone
        self.vector_store.add_documents(all_chunks, self.namespace)
        print(f"✅ Documents added to Pinecone index")
    
    def query_existing_documents(self, 
                                question: str, 
                                max_context_length: int = 4000, 
                                k: int = 5,
                                filter: Dict = None) -> str:
        """Query existing documents in Pinecone (no need to add new documents)"""
        print(f"Querying existing Pinecone documents: {question}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Search Pinecone
        relevant_docs = self.vector_store.similarity_search(
            query_embedding, k=k, namespace=self.namespace, filter=filter
        )
        
        if not relevant_docs:
            return "No relevant documents found in Pinecone index."
        
        # Build context from relevant documents
        context_parts = []
        current_length = 0
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        print(f"Found {len(relevant_docs)} relevant documents:")
        for i, (doc, score) in enumerate(relevant_docs):
            print(f"  {i+1}. Score: {score:.4f}, Source: {doc.source}")
            
            doc_tokens = len(tokenizer.encode(doc.content))
            if current_length + doc_tokens <= max_context_length:
                context_parts.append(f"Source: {doc.source}\n{doc.content}")
                current_length += doc_tokens
            else:
                break
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""Based on the following context from the document database, please answer the question. If the context doesn't contain enough information to answer the question completely, please say so and provide what information is available.

Context:
{context}

Question: {question}

Answer:"""

        try:
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Handle the response content properly
            content = response.content[0]
            if hasattr(content, 'text'):
                return content.text
            else:
                return str(content)
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        pinecone_stats = self.vector_store.get_index_stats()
        
        return {
            "pinecone_index": self.vector_store.index_name,
            "namespace": self.namespace,
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dimension": self.embedding_generator.embedding_dimension,
            **pinecone_stats
        }

# Example usage
def main():
    """Example usage of Claude RAG with Pinecone"""
    
    # Initialize RAG system with Pinecone
    # Make sure you have ANTHROPIC_API_KEY and PINECONE_API_KEY in your environment
    try:
        rag = ClaudeRAGPinecone(
            index_name="claude-rag-demo",  # Use your existing index name
            namespace="demo"  # Optional namespace
        )
        
        # Option 1: Query existing documents (if you already have documents in Pinecone)
        print("\n" + "="*60)
        print("QUERYING EXISTING DOCUMENTS IN PINECONE")
        print("="*60)
        
        questions = [
            "What documents do you have?",
            "Tell me about artificial intelligence",
            "How does machine learning work?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            print("-" * 40)
            answer = rag.query_existing_documents(question)
            print(f"A: {answer}")
        
        # Option 2: Add new documents (if you want to add more)
        print("\n" + "="*60)
        print("ADDING NEW DOCUMENTS TO PINECONE")
        print("="*60)
        
        new_documents = [
            """
            Computer Vision is a field of artificial intelligence that enables computers 
            to interpret and understand visual information from the world. It involves 
            developing algorithms and techniques that can automatically extract, analyze, 
            and understand useful information from images, videos, and other visual inputs.
            """
        ]
        
        rag.add_documents(new_documents, ["CV_Guide"], ["cv_doc"])
        
        # Query after adding new documents
        answer = rag.query_existing_documents("What is computer vision?")
        print(f"\nAfter adding new document:")
        print(f"Q: What is computer vision?")
        print(f"A: {answer}")
        
        # Print stats
        stats = rag.get_stats()
        print(f"\n📊 RAG System Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. ANTHROPIC_API_KEY in your .env file")
        print("2. PINECONE_API_KEY in your .env file")
        print("3. Pinecone client installed: pip install pinecone-client")

if __name__ == "__main__":
    main() 