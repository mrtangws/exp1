#!/usr/bin/env python3
"""
Script to ingest documents into Pinecone for RAG
"""

import os
from claude_rag_pinecone import ClaudeRAGPinecone

def ingest_text_files(folder_path, index_name="claude-rag", namespace="", no_chunking=False):
    """Ingest all text files from a folder"""
    rag = ClaudeRAGPinecone(
        index_name=index_name,
        namespace=namespace,
        embedding_model="sentence-transformers/all-roberta-large-v1"
    )
    
    documents = []
    sources = []
    doc_ids = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.txt', '.csv')):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                sources.append(filename)
                doc_ids.append(filename.replace('.', '_'))
    
    if documents:
        if no_chunking:
            from claude_rag_pinecone import Document
            docs = [Document(content=text, source=source, doc_id=doc_id, metadata={"source": source}) 
                   for text, source, doc_id in zip(documents, sources, doc_ids)]
            embeddings = rag.embedding_generator.generate_embeddings_batch([doc.content for doc in docs])
            for doc, embedding in zip(docs, embeddings):
                doc.embedding = embedding
            rag.vector_store.add_documents(docs, rag.namespace)
        else:
            rag.add_documents(documents, sources, doc_ids)
        print(f"Ingested {len(documents)} documents {'without chunking' if no_chunking else 'with chunking'}")
    else:
        print("No text files found")

def ingest_manual_documents(index_name="claude-rag", namespace=""):
    """Manually add documents"""
    rag = ClaudeRAGPinecone(
        index_name=index_name,
        namespace=namespace,
        embedding_model="sentence-transformers/all-roberta-large-v1"
    )
    
    documents = [
        "Your document content here...",
        "Another document...",
    ]
    
    sources = ["doc1", "doc2"]
    doc_ids = ["manual_doc_1", "manual_doc_2"]
    
    rag.add_documents(documents, sources, doc_ids)
    print(f"Added {len(documents)} documents")

if __name__ == "__main__":
    # Option 1: Ingest from folder with chunking
    # ingest_text_files("./documents", "cfo-python", "cfo")
    
    # Option 2: Ingest from folder without chunking
    ingest_text_files("./documents", "cfo-python", "cfo", no_chunking=True)
    
    # Option 3: Manual ingestion
    # ingest_manual_documents("cfo-vector", "cfo")