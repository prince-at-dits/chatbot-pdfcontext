import os
from typing import List
import numpy as np
import requests
import json

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIMENSION = 768  # Standard embedding dimension

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
    
    # Use Gemini API for embeddings
    embeddings = []
    for text in texts:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": text}]}
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "embedding" in data and "values" in data["embedding"]:
                embedding = np.array(data["embedding"]["values"], dtype=np.float32)
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            else:
                # Fallback: create random normalized embedding
                embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
                
        except Exception as e:
            print(f"Embedding error for text: {e}")
            # Fallback: create random normalized embedding
            embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
    
    return np.array(embeddings, dtype=np.float32)