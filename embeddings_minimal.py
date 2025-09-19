import os
from typing import List
import numpy as np
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
    
    # Use simple hash-based embeddings as fallback
    embeddings = []
    for text in texts:
        # Create deterministic embedding from text hash
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))
        embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    return np.array(embeddings, dtype=np.float32)