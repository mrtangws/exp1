#!/usr/bin/env python3
"""
Ingest documents into Pinecone without chunking
"""

import os
from claude_rag_pinecone import ClaudeRAGPinecone, LocalEmbeddingGenerator, Document

def add_documents_no_chunking(rag, texts, sources=None, doc_ids=None):
    """Add documents without chunking"""
    if sources is None:
        sources = [f"document_{i}" for i in range(len(texts))]
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(len(texts))]
    
    # Create documents without chunking
    documents = []
    for text, source, doc_id in zip(texts, sources, doc_ids):
        doc = Document(
            content=text,
            source=source,
            doc_id=doc_id,
            metadata={"source": source}
        )
        documents.append(doc)
    
    # Generate embeddings
    embeddings = rag.embedding_generator.generate_embeddings_batch([doc.content for doc in documents])
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc.embedding = embedding
    
    # Add to Pinecone
    rag.vector_store.add_documents(documents, rag.namespace)

def main():
    rag = ClaudeRAGPinecone(
        index_name="cfo-vector",
        namespace="cfo",
        embedding_model="sentence-transformers/all-roberta-large-v1"
    )
    
    # Your full documents (no chunking)
    documents = [
        "Complete document 1 content here...",
        "Complete document 2 content here..."
    ]
    
    sources = ["doc1.txt", "doc2.txt"]
    doc_ids = ["full_doc_1", "full_doc_2"]
    
    add_documents_no_chunking(rag, documents, sources, doc_ids)
    print(f"Added {len(documents)} documents without chunking")

if __name__ == "__main__":
    main()