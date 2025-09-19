import os
from typing import List
import numpy as np
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

# Simple vocabulary for TF-IDF style embeddings
GLOBAL_VOCAB = {}
VOCAB_SIZE = 1000

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
    
    global GLOBAL_VOCAB
    
    # Build vocabulary from all texts
    all_words = set()
    for text in texts:
        words = text.lower().split()
        all_words.update(words)
    
    # Update global vocabulary
    for word in all_words:
        if word not in GLOBAL_VOCAB and len(GLOBAL_VOCAB) < VOCAB_SIZE:
            GLOBAL_VOCAB[word] = len(GLOBAL_VOCAB)
    
    # Create embeddings based on word presence
    embeddings = []
    for text in texts:
        words = text.lower().split()
        embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        
        # Set dimensions based on word presence
        for word in words:
            if word in GLOBAL_VOCAB:
                idx = GLOBAL_VOCAB[word] % EMBEDDING_DIMENSION
                embedding[idx] += 1.0
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        else:
            # Fallback for empty embeddings
            embedding[0] = 1.0
            
        embeddings.append(embedding)
    
    return np.array(embeddings, dtype=np.float32)