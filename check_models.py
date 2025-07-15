#!/usr/bin/env python3
"""
Check available embedding models and their dimensions
"""

from sentence_transformers import SentenceTransformer

# Models that might produce 1024 dimensions
models_to_check = [
    "all-mpnet-base-v2",  # 768 dimensions
    "all-MiniLM-L12-v2",  # 384 dimensions  
    "paraphrase-multilingual-mpnet-base-v2",  # 768 dimensions
    "sentence-transformers/all-roberta-large-v1",  # 1024 dimensions
]

for model_name in models_to_check:
    try:
        print(f"\nTesting model: {model_name}")
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        print(f"  Dimensions: {dim}")
        
        if dim == 1024:
            print(f"  This model matches your index dimension!")
            break
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")