#!/usr/bin/env python3
"""
Debug script to check what context is being sent to Claude
"""

import os
from claude_rag_pinecone import ClaudeRAGPinecone

try:
    rag = ClaudeRAGPinecone(
        index_name="cfo-python",
        namespace="cfo",
        embedding_model="sentence-transformers/all-roberta-large-v1"
    )
    
    # Test query
    question = "tell me about july appointment"
    query_embedding = rag.embedding_generator.generate_embedding(question)
    
    # Search Pinecone
    relevant_docs = rag.vector_store.similarity_search(
        query_embedding, k=3, namespace=rag.namespace
    )
    
    print(f"Found {len(relevant_docs)} documents:")
    for i, (doc, score) in enumerate(relevant_docs):
        print(f"\nDocument {i+1}:")
        print(f"  Score: {score:.4f}")
        print(f"  Source: {doc.source}")
        print(f"  Content length: {len(doc.content)}")
        print(f"  Content preview: {doc.content[:200]}...")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()