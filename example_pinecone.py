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
        print("Error: ANTHROPIC_API_KEY not found")
        print("Please add your Anthropic API key to .env file")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found")
        print("Please add your Pinecone API key to .env file")
        return
    
    # Configuration - Update these for your setup
    INDEX_NAME = input("Enter your Pinecone index name (or press Enter for 'cfo-python'): ").strip()
    if not INDEX_NAME:
        INDEX_NAME = "cfo-python"
    
    NAMESPACE = input("Enter namespace (optional, press Enter for default): ").strip()
    if not NAMESPACE:
        NAMESPACE = "cfo"
    
    print(f"\nConnecting to Pinecone...")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Namespace: {NAMESPACE or 'default'}")
    
    try:
        # Initialize RAG system with your Pinecone index
        rag = ClaudeRAGPinecone(
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
            embedding_model="sentence-transformers/all-roberta-large-v1",  # 1024 dimensions to match your index
            claude_model="claude-3-5-sonnet-20241022",
            use_openai_embeddings=False  # Use local embeddings for 1024 dimensions
        )
        
        # Get index statistics
        stats = rag.get_stats()
        print(f"\nYour Pinecone Index Stats:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        if stats.get('total_vectors', 0) == 0:
            print("\n⚠️  Your index appears to be empty.")
            print("Would you like to add some sample documents? (y/n)")
            add_sample = input().strip().lower()
            
            if add_sample == 'y':
                print("\nAdding sample documents...")
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
                print("Sample documents added!")
        
        # Interactive query session
        print("\n" + "="*60)
        print("CLAUDE RAG with PINECONE - Ask questions about your documents!")
        print("="*60)
        print("Tips:")
        print("   • Ask specific questions about your document content")
        print("   • Try asking for summaries or comparisons")
        print("   • Use 'stats' to see index information")
        print("   • Type 'quit' to exit")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = rag.get_stats()
                    print("\nCurrent Index Stats:")
                    for key, value in stats.items():
                        print(f"   • {key}: {value}")
                    continue
                
                if not query:
                    continue
                
                print("Searching Pinecone and generating response...")
                
                # Query with additional options
                system_instructions = """You are a clinic assistant providing quick information retrieval for clinic staff on patients, treatments, appointments, inventory and payables & receivables.

1. When given a string of cancer code and treatment codes:
  1a. Return a list of invoice items with quantity, 
  1b. Use the items to fetch the selling_price, then return the sum total.
2. When asked about cashflow, revenue and expenses, compare month to month data and analyze.

Note: 
-  If a query requires you to know the date, use the action group today_date to retrieve today's date.
- Date format in knowledge base is YYYY-MM-DD
- A week starts on Monday and ends on Sunday. You can ignore Saturday and Sunday when answering queries, as the clinic is closed. 
- Low stock means the stock count is less than 10. Please omit items with stock count <= 0.
- When listing multiple items, always use Unicode Character “•” (U+2022) and add a newline (\r\n) after each item.
- Revenue == Cash Inflow, Expense == Cash Outflow.

Here are some of the common abbreviation we use:
- appt: appointment"""
                
                answer = rag.query_existing_documents(
                    query, 
                    k=10,  # Number of similar documents to retrieve
                    max_context_length=4000,  # Max tokens for context
                    system_instructions=system_instructions
                )
                
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
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
    print("Claude RAG + Pinecone Integration")
    print("This example connects to your existing Pinecone documents")
    
    main()
    
    # Uncomment to run advanced example
    # advanced_example() 