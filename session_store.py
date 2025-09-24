import threading
from typing import List, Dict


class SessionIndex:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.lock = threading.RLock()
        # Track distinct filenames uploaded in this session to surface to the user
        self.uploaded_filenames: Dict[str, None] = {}
        # LangChain-based state (mirrors teammate's pipeline)
        self.lc_pipeline = None  # Single pipeline object encapsulating retriever/QA/entities/metadata


    def add_to_conversation(self, role: str, content: str):
        if not content or not content.strip():
            return
        with self.lock:
            self.conversation_history.append({"role": role, "content": content.strip()})
            if len(self.conversation_history) > 16:
                self.conversation_history = self.conversation_history[-16:]

    def get_conversation_context(self, max_turns: int = 4) -> str:
        with self.lock:
            if not self.conversation_history:
                return ""
            recent = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
            parts: List[str] = []
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get('content', '').strip()
                if content:
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)

    def get_uploaded_filenames(self) -> List[str]:
        # Return a stable, de-duplicated list of filenames
        with self.lock:
            return sorted(self.uploaded_filenames.keys())

    def get_pages_by_file(self) -> Dict[str, int]:
        with self.lock:
            if self.lc_pipeline and self.lc_pipeline.metadata:
                return {self.lc_pipeline.metadata.get("name", "unknown"): self.lc_pipeline.metadata.get("num_pages", 0)}
            return {}

    def get_total_pages(self) -> int:
        with self.lock:
            if self.lc_pipeline and self.lc_pipeline.metadata:
                return self.lc_pipeline.metadata.get("num_pages", 0)
            return 0


