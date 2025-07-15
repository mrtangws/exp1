#!/usr/bin/env python3
"""
Test July appointment query
"""

import os
from claude_rag_pinecone import ClaudeRAGPinecone

try:
    rag = ClaudeRAGPinecone(
        index_name="cfo-python",
        namespace="cfo",
        embedding_model="sentence-transformers/all-roberta-large-v1"
    )
    
    answer = rag.query_existing_documents(
        "tell me about july appointment",
        k=3,
        system_instructions="You are a clinic assistant. Analyze the appointment data and provide specific information about July appointments."
    )
    
    print(f"Answer: {answer}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()