import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBEDDING_DIMENSION = _embedding_model.get_sentence_embedding_dimension()


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
    vectors = _embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


