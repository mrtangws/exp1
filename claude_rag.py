import os
import re
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Document:
    """Represents a document chunk with its metadata"""
    content: str
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

class DocumentProcessor:
    """Handles document chunking and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def chunk_text(self, text: str, source: str = "") -> List[Document]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunks.append(Document(
                    content=current_chunk.strip(),
                    metadata={"tokens": current_tokens, "source": source}
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.tokenizer.encode(current_chunk))
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Document(
                content=current_chunk.strip(),
                metadata={"tokens": current_tokens, "source": source}
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.tokenizer.decode(overlap_tokens)

class EmbeddingGenerator:
    """Handles text embedding generation using OpenAI"""
    
    def __init__(self, api_key: str = None):
        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=api_key_to_use)
        self.model = "text-embedding-ada-002"
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(1536)  # Default embedding size for ada-002
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        batch_size = 100  # OpenAI API limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating batch embeddings: {e}")
                # Fallback to individual generation
                for text in batch:
                    embeddings.append(self.generate_embedding(text))
        
        return embeddings

class VectorStore:
    """Simple in-memory vector store for document embeddings"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        self.documents.extend(documents)
        
        # Combine all embeddings
        embeddings = [doc.embedding for doc in documents if doc.embedding is not None]
        if embeddings:
            new_embeddings = np.vstack(embeddings)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Find k most similar documents to query"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return documents with similarity scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((self.documents[idx], similarities[idx]))
        
        return results

class ClaudeRAG:
    """Main RAG system using Claude API"""
    
    def __init__(self, anthropic_api_key: str = None, openai_api_key: str = None):
        anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("Anthropic API key is required")
        
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        
        self.processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator(openai_api_key)
        self.vector_store = VectorStore()
    
    def add_documents(self, texts: List[str], sources: List[str] = None) -> None:
        """Add documents to the RAG system"""
        if sources is None:
            sources = [f"document_{i}" for i in range(len(texts))]
        
        print(f"Processing {len(texts)} documents...")
        
        # Chunk documents
        all_chunks = []
        for text, source in zip(texts, sources):
            chunks = self.processor.chunk_text(text, source)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks)
        print(f"Added documents to vector store. Total documents: {len(self.vector_store.documents)}")
    
    def query(self, question: str, max_context_length: int = 4000, k: int = 5) -> str:
        """Query the RAG system"""
        print(f"Processing query: {question}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query_embedding, k=k)
        
        if not relevant_docs:
            return "No relevant documents found."
        
        # Build context from relevant documents
        context_parts = []
        current_length = 0
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        for doc, similarity in relevant_docs:
            doc_tokens = len(tokenizer.encode(doc.content))
            if current_length + doc_tokens <= max_context_length:
                context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.content}")
                current_length += doc_tokens
            else:
                break
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question completely, please say so and provide what information is available.

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
            
            return response.content[0].text
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "total_documents": len(self.vector_store.documents),
            "embedding_dimension": self.vector_store.embeddings.shape[1] if self.vector_store.embeddings is not None else 0,
            "unique_sources": len(set(doc.metadata.get('source', '') for doc in self.vector_store.documents))
        }

# Example usage
def main():
    """Example usage of the Claude RAG system"""
    
    # Initialize RAG system
    rag = ClaudeRAG()
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that work and react like humans. Some of the activities 
        computers with artificial intelligence are designed for include speech recognition, 
        learning, planning, and problem solving. AI research has been highly successful 
        in developing effective techniques for solving a wide range of problems, from game 
        playing to medical diagnosis to logistics planning.
        """,
        """
        Machine Learning is a subset of artificial intelligence (AI) that provides systems 
        the ability to automatically learn and improve from experience without being 
        explicitly programmed. Machine learning focuses on the development of computer 
        programs that can access data and use it to learn for themselves. The process of 
        learning begins with observations or data, such as examples, direct experience, 
        or instruction, in order to look for patterns in data.
        """,
        """
        Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and 
        human language, in particular how to program computers to process and analyze 
        large amounts of natural language data. NLP combines computational linguistics 
        with statistical, machine learning, and deep learning models.
        """
    ]
    
    sources = ["AI_Overview", "ML_Basics", "NLP_Introduction"]
    
    # Add documents to RAG system
    rag.add_documents(documents, sources)
    
    # Print system stats
    stats = rag.get_stats()
    print(f"\nRAG System Stats: {stats}")
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is the relationship between AI and NLP?",
        "Tell me about computer vision"  # This should indicate limited context
    ]
    
    print("\n" + "="*50)
    print("EXAMPLE QUERIES")
    print("="*50)
    
    for query in queries:
        print(f"\nQ: {query}")
        print("-" * 30)
        answer = rag.query(query)
        print(f"A: {answer}")
        print()

if __name__ == "__main__":
    main() 