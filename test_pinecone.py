#!/usr/bin/env python3
"""
Simple test script to verify Pinecone connection
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from claude_rag_pinecone import ClaudeRAGPinecone
    
    print("Testing Pinecone connection...")
    
    # Test with your specific index and namespace
    rag = ClaudeRAGPinecone(
        index_name="cfo-vector",
        namespace="cfo",
        embedding_model="sentence-transformers/all-roberta-large-v1"  # 1024 dimensions
    )
    
    print("Connection successful!")
    
    # Get stats
    stats = rag.get_stats()
    print("\nIndex Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test a simple query
    print("\nTesting query...")
    answer = rag.query_existing_documents("How many appointments are scheduled?", k=3)
    print(f"Query result: {answer}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()