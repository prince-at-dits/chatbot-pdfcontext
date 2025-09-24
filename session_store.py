import threading
import re
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
        # Use a reentrant lock because some helper methods acquire the lock
        # from within operations that already hold it (e.g., add -> _update_extracted_fields)
        self.lock = threading.RLock()
        # Track distinct filenames uploaded in this session to surface to the user
        self.uploaded_filenames: Dict[str, None] = {}
        # Store extracted structured fields from uploaded resume-like PDFs
        self.extracted_fields: Dict[str, str] = {}
        # Track pages per uploaded file
        self.filename_to_pages: Dict[str, set] = {}
        # LangChain-based state (mirrors teammate's pipeline)
        self.lc_pipeline = None  # Single pipeline object encapsulating retriever/QA/entities/metadata

    def add(self, embeddings: np.ndarray, chunks: List[str], metadata: List[Dict] = None):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        with self.lock:
            self.text_chunks.extend(chunks)
            if metadata:
                self.chunk_metadata.extend(metadata)
                # Record filenames present in metadata
                for m in metadata:
                    fname = m.get('filename') if isinstance(m, dict) else None
                    if fname:
                        self.uploaded_filenames[fname] = None
                        # Track page numbers per file if available
                        pg = m.get('page') if isinstance(m, dict) else None
                        if pg is not None:
                            try:
                                pgi = int(pg)
                            except Exception:
                                continue
                            if fname not in self.filename_to_pages:
                                self.filename_to_pages[fname] = set()
                            self.filename_to_pages[fname].add(pgi)
            else:
                self.chunk_metadata.extend([{}] * len(chunks))
            self.index.add(embeddings)
            # Update extracted fields based on accumulated text
            self._update_extracted_fields()

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
            return {fn: len(pgs) for fn, pgs in self.filename_to_pages.items()}

    def get_total_pages(self) -> int:
        with self.lock:
            return sum(len(pgs) for pgs in self.filename_to_pages.values())

    def _get_all_text(self) -> str:
        with self.lock:
            return "\n".join(self.text_chunks)

    def _update_extracted_fields(self) -> None:
        text = self._get_all_text()
        if not text:
            return
        # Basic regex patterns
        email_matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        phone_matches = re.findall(r"(?:\+\d{1,3}[-\s]?)?\d{3,4}[-\s]?\d{3}[-\s]?\d{3,4}", text)
        linkedin = re.search(r"linkedin\.com\/in\/\S+", text, re.IGNORECASE)
        github = re.search(r"github\.com\/\S+", text, re.IGNORECASE)
        portfolio = re.search(r"https?:\/\/\S+vercel\.app\S*", text, re.IGNORECASE)

        # Heuristic name extraction: first line with two capitalized tokens before contact
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        candidate_name = ""
        if lines:
            # Try first 5 lines for a person-like name
            for l in lines[:5]:
                # Skip lines that are likely contact lines
                if re.search(r"@|\b(\+?\d{3,}|http|www\.)\b", l, re.IGNORECASE):
                    continue
                tokens = l.split()
                caps = [t for t in tokens if re.match(r"^[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?$", t)]
                if len(caps) >= 2:
                    candidate_name = " ".join(caps[:3])
                    break

        # Skills extraction: find 'Skills' section line and take following comma-separated
        skills_list: List[str] = []
        for i, l in enumerate(lines):
            if re.match(r"^Skills\b", l, re.IGNORECASE):
                # Use this line and maybe next line if continuation
                skill_text = l
                if i + 1 < len(lines) and len(lines[i + 1].split(',')) > 1:
                    skill_text += ", " + lines[i + 1]
                # Remove 'Skills'
                skill_text = re.sub(r"^Skills\s*[:\-]?\s*", "", skill_text, flags=re.IGNORECASE)
                parts = [s.strip() for s in re.split(r",|•|\|", skill_text) if s.strip()]
                # De-duplicate preserve order
                seen = set()
                for p in parts:
                    if p.lower() not in seen:
                        seen.add(p.lower())
                        skills_list.append(p)
                break

        # Role/title extraction: look for 'Full Stack', 'Developer', 'Engineer' near top
        title = ""
        for l in lines[:8]:
            if re.search(r"(Full\s*Stack|Developer|Engineer|Software|DevOps)", l, re.IGNORECASE):
                title = l
                break

        # Experience extraction (very light): company lines with dates
        experience_lines: List[str] = []
        for l in lines:
            if re.search(r"\b(\d{2}\/\d{4})\s*[–-]\s*(\d{2}\/\d{4}|Present)\b", l):
                experience_lines.append(l)

        with self.lock:
            if candidate_name:
                self.extracted_fields['name'] = candidate_name
            if email_matches:
                self.extracted_fields['email'] = email_matches[0]
            if phone_matches:
                self.extracted_fields['phone'] = phone_matches[0]
            if linkedin:
                self.extracted_fields['linkedin'] = linkedin.group(0)
            if github:
                self.extracted_fields['github'] = github.group(0)
            if portfolio:
                self.extracted_fields['portfolio'] = portfolio.group(0)
            if skills_list:
                self.extracted_fields['skills'] = ", ".join(skills_list)
            if title:
                self.extracted_fields['title'] = title
            if experience_lines:
                self.extracted_fields['experience_summary'] = "; ".join(experience_lines[:3])

    def get_extracted(self) -> Dict[str, str]:
        with self.lock:
            return dict(self.extracted_fields)

    def get_representative_contexts(self, top_n: int = 6) -> List[Tuple[str, float, Dict]]:
        """Pick evenly spaced chunks across the document to represent overall content."""
        with self.lock:
            if not self.text_chunks:
                return []
            n = len(self.text_chunks)
            if top_n >= n:
                return [(self.text_chunks[i], 1.0, self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}) for i in range(n)]
            # Evenly spaced indices
            indices = [int(round(i * (n - 1) / max(1, top_n - 1))) for i in range(top_n)]
            results: List[Tuple[str, float, Dict]] = []
            for i in indices:
                meta = self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}
                results.append((self.text_chunks[i], 1.0, meta))
            return results

    def get_representative_by_pages(self, max_per_page: int = 1, limit: int = 12) -> List[Tuple[str, float, Dict]]:
        """Sample the first chunk(s) per page to cover the whole document."""
        with self.lock:
            page_to_indices: Dict[int, List[int]] = {}
            for idx, meta in enumerate(self.chunk_metadata):
                if not isinstance(meta, dict):
                    continue
                if 'page' in meta:
                    try:
                        p = int(meta['page'])
                    except Exception:
                        continue
                    page_to_indices.setdefault(p, []).append(idx)
            results: List[Tuple[str, float, Dict]] = []
            for p in sorted(page_to_indices.keys()):
                for idx in page_to_indices[p][:max_per_page]:
                    meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                    results.append((self.text_chunks[idx], 1.0, meta))
                    if len(results) >= limit:
                        return results
            return results


