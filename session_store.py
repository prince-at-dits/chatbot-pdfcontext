import threading
from typing import List, Dict, Tuple

import numpy as np
import faiss


class SessionIndex:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.text_chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.lock = threading.Lock()

    def add(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict] = None):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        with self.lock:
            self.text_chunks.extend(chunks)
            if metadata:
                self.chunk_metadata.extend(metadata)
            else:
                self.chunk_metadata.extend([{}] * len(chunks))
            self.index.add(embeddings)

    def search(self, query_emb: np.ndarray, top_k: int = 8, min_score: float = 0.1) -> List[Tuple[str, float, Dict]]:
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)
        with self.lock:
            if self.index.ntotal == 0:
                return []
            scores, indices = self.index.search(query_emb, min(top_k * 2, self.index.ntotal))
        results: List[Tuple[str, float, Dict]] = []
        for i in range(indices.shape[1]):
            idx = indices[0, i]
            if idx == -1:
                continue
            text = self.text_chunks[idx]
            score = float(scores[0, i])
            if score >= min_score:
                metadata = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                results.append((text, score, metadata))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def add_to_conversation(self, role: str, content: str):
        with self.lock:
            self.conversation_history.append({"role": role, "content": content})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

    def get_conversation_context(self, max_turns: int = 6) -> str:
        with self.lock:
            recent = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
            parts: List[str] = []
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                parts.append(f"{role}: {msg.get('content','')}")
            return "\n".join(parts)


