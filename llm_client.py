import os
import json
from typing import List, Tuple, Dict

import requests


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def generate_with_ollama(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    if not GEMINI_API_KEY:
        return "[Error] GEMINI_API_KEY not set"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Combine system and user prompts for Gemini
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    payload = {
        "contents": [{
            "parts": [{"text": combined_prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"].strip()
        
        return "[Error] No response generated"
    except Exception as e:
        error_msg = str(e)
        # Mask API key in error messages
        if GEMINI_API_KEY and GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(GEMINI_API_KEY, "***MASKED***")
        return f"[LLM error] {error_msg}"


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


