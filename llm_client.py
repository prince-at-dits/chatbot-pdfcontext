import os
import json
from typing import List, Tuple, Dict

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def generate_with_ollama(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    url = f"{OLLAMA_URL}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "num_thread": 20
        }
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            msg = data.get("message", {})
            content = msg.get("content") or data.get("response")
            if content:
                return content.strip()
        return ""
    except Exception as e:
        return f"[LLM error] {e}"


def build_rag_prompt(question: str, contexts: List[Tuple[str, float, Dict]], conversation_context: str = "") -> str:
    context_texts: List[str] = []
    for i, (text, score, metadata) in enumerate(contexts, 1):
        page_info = f" (Page {metadata.get('page', '?')})" if metadata.get('page') else ""
        context_texts.append(f"[Context {i}{page_info}]: {text}")
    joined = "\n\n".join(context_texts)
    conversation_part = f"\n\nPrevious conversation:\n{conversation_context}\n" if conversation_context.strip() else ""
    prompt = f"""You are a helpful AI assistant with access to document context. Your task is to answer questions accurately based on the provided context.

Instructions:
1. Answer the user's question using ONLY the information from the provided context
2. If the answer is not in the context, clearly state "I don't have enough information in the provided context to answer this question"
3. Be specific and cite relevant details from the context
4. If you're unsure, say so rather than guessing
5. Maintain a helpful and professional tone

Context from documents:
{joined}{conversation_part}

Question: {question}

Answer:"""
    return prompt


