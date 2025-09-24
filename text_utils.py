import io
import re
from typing import List, Tuple, Dict

from PyPDF2 import PdfReader


def chunk_text(text: str, max_chars: int = 500, overlap: int = 100) -> List[str]:
    """Sentence-aware chunking with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c.strip()) > 30]


def pdf_to_text_chunks(pdf_bytes: bytes) -> List[Tuple[str, Dict]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks_with_metadata: List[Tuple[str, Dict]] = []
    for page_num, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            txt = re.sub(r'\s+', ' ', txt.strip())
            for chunk in chunk_text(txt):
                metadata = {"page": page_num, "source": "pdf", "chunk_length": len(chunk)}
                chunks_with_metadata.append((chunk, metadata))
    return chunks_with_metadata


