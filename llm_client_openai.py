import os
import json
from typing import List, Tuple, Dict
import requests

def generate_with_openai(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[Error] OPENAI_API_KEY not set"
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error] {e}"

def build_rag_prompt(question: str, contexts: List[Tuple[str, float, Dict]], conversation_context: str = "") -> str:
    context_texts: List[str] = []
    for i, (text, score, metadata) in enumerate(contexts, 1):
        page_info = f" (Page {metadata.get('page', '?')})" if metadata.get('page') else ""
        context_texts.append(f"[Context {i}{page_info}]: {text}")
    joined = "\n\n".join(context_texts)
    conversation_part = f"\n\nPrevious conversation:\n{conversation_context}\n" if conversation_context.strip() else ""
    prompt = f"""Answer the question using ONLY the provided context.

Context:
{joined}{conversation_part}

Question: {question}

Answer:"""
    return prompt