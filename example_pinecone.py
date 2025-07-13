#!/usr/bin/env python3
"""
Example: Using Claude RAG with existing Pinecone documents.

This example shows how to connect to your existing Pinecone index and query
your documents using Claude for intelligent responses.
"""

import os
from claude_rag_pinecone import ClaudeRAGPinecone

def main():
    """Main example function"""
    
    # Check required API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        print("Please add your Anthropic API key to .env file")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not found")
        print("Please add your Pinecone API key to .env file")
        return
    
    # Configuration - Update these for your setup
    INDEX_NAME = input("Enter your Pinecone index name (or press Enter for 'claude-rag'): ").strip()
    if not INDEX_NAME:
        INDEX_NAME = "claude-rag"
    
    NAMESPACE = input("Enter namespace (optional, press Enter for default): ").strip()
    
    print(f"\n🚀 Connecting to Pinecone...")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Namespace: {NAMESPACE or 'default'}")
    
    try:
        # Initialize RAG system with your Pinecone index
        rag = ClaudeRAGPinecone(
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
            embedding_model="all-MiniLM-L6-v2"  # Use local embeddings
        )
        
        # Get index statistics
        stats = rag.get_stats()
        print(f"\n📊 Your Pinecone Index Stats:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        if stats.get('total_vectors', 0) == 0:
            print("\n⚠️  Your index appears to be empty.")
            print("Would you like to add some sample documents? (y/n)")
            add_sample = input().strip().lower()
            
            if add_sample == 'y':
                print("\n📚 Adding sample documents...")
                sample_docs = [
                    """
                    Artificial Intelligence (AI) is transforming industries worldwide. From healthcare 
                    to finance, AI systems are helping automate complex tasks, provide insights from 
                    large datasets, and enable new capabilities that were previously impossible. 
                    Machine learning, a subset of AI, allows systems to learn and improve from 
                    experience without being explicitly programmed.
                    """,
                    """
                    Python is one of the most popular programming languages for data science and 
                    machine learning. Its simplicity and extensive library ecosystem make it ideal 
                    for rapid prototyping and development. Key libraries include NumPy for numerical 
                    computing, Pandas for data manipulation, and scikit-learn for machine learning.
                    """,
                    """
                    Vector databases are specialized databases designed to store and query 
                    high-dimensional vectors efficiently. They're essential for AI applications 
                    like similarity search, recommendation systems, and retrieval-augmented 
                    generation (RAG). Popular vector databases include Pinecone, Weaviate, and Chroma.
                    """
                ]
                
                rag.add_documents(
                    sample_docs, 
                    sources=["AI_Overview", "Python_Guide", "Vector_DB_Guide"],
                    doc_id_prefixes=["ai_doc", "python_doc", "vectordb_doc"]
                )
                print("✅ Sample documents added!")
        
        # Interactive query session
        print("\n" + "="*60)
        print("🤖 CLAUDE RAG with PINECONE - Ask questions about your documents!")
        print("="*60)
        print("💡 Tips:")
        print("   • Ask specific questions about your document content")
        print("   • Try asking for summaries or comparisons")
        print("   • Use 'stats' to see index information")
        print("   • Type 'quit' to exit")
        
        while True:
            try:
                query = input("\n🔍 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = rag.get_stats()
                    print("\n📊 Current Index Stats:")
                    for key, value in stats.items():
                        print(f"   • {key}: {value}")
                    continue
                
                if not query:
                    continue
                
                print("🔎 Searching Pinecone and generating response...")
                
                # Query with additional options
                answer = rag.query_existing_documents(
                    query, 
                    k=5,  # Number of similar documents to retrieve
                    max_context_length=4000  # Max tokens for context
                )
                
                print(f"\n🤖 Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        print("\nTroubleshooting:")
        print("1. Check your PINECONE_API_KEY in .env")
        print("2. Verify your index name exists in Pinecone")
        print("3. Make sure you have the pinecone-client installed:")
        print("   pip install pinecone-client")

def advanced_example():
    """Advanced example with filtering and metadata"""
    
    print("\n" + "="*60)
    print("ADVANCED EXAMPLE: Filtering and Metadata")
    print("="*60)
    
    try:
        rag = ClaudeRAGPinecone(
            index_name="your-index-name",  # Replace with your index
            namespace="your-namespace"     # Replace with your namespace
        )
        
        # Example: Query with metadata filtering
        # This filters results to only documents from a specific source
        answer = rag.query_existing_documents(
            "Tell me about machine learning",
            k=3,
            filter={"source": {"$eq": "ml_textbook"}}  # Only from ml_textbook source
        )
        print(f"Filtered answer: {answer}")
        
        # Example: Query different namespaces
        # You can organize documents in different namespaces
        rag.namespace = "technical_docs"
        answer = rag.query_existing_documents("What is Python?")
        print(f"From technical_docs namespace: {answer}")
        
    except Exception as e:
        print(f"Advanced example error: {e}")

if __name__ == "__main__":
    print("🎯 Claude RAG + Pinecone Integration")
    print("This example connects to your existing Pinecone documents")
    
    main()
    
    # Uncomment to run advanced example
    # advanced_example() 