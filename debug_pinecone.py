#!/usr/bin/env python3
"""
Debug script to inspect Pinecone metadata structure
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

try:
    # Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("cfo-vector")
    
    # Query to get some vectors with metadata
    response = index.query(
        vector=[0.0] * 1024,  # Dummy vector
        top_k=3,
        include_metadata=True,
        namespace="cfo"
    )
    
    print("Raw Pinecone response:")
    print("=" * 50)
    
    for i, match in enumerate(response.matches):
        print(f"\nDocument {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Score: {match.score}")
        print(f"  Metadata keys: {list(match.metadata.keys())}")
        print(f"  Full metadata: {match.metadata}")
        
        # Check for common content field names
        content_fields = ['content', 'text', 'chunk', 'document', 'body', 'data']
        for field in content_fields:
            if field in match.metadata:
                print(f"  Found content in '{field}': {match.metadata[field][:100]}...")
                break
        else:
            print("  No content found in common fields")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()