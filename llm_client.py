import os
from typing import List, Tuple, Dict

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

print(f"DEBUG: OPENAI_API_KEY loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"DEBUG: Using model: {OPENAI_MODEL}")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_with_openai(system_prompt: str, user_prompt: str, timeout: int = 45) -> str:
    if not OPENAI_API_KEY or not client:
        return f"[Error] OPENAI_API_KEY not set. Env vars: {list(os.environ.keys())[:5]}"
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=450,
            timeout=timeout
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()
        
        return "[Error] No response generated"
    except Exception as e:
        error_msg = str(e)
        # Mask API key in error messages
        if OPENAI_API_KEY and OPENAI_API_KEY in error_msg:
            error_msg = error_msg.replace(OPENAI_API_KEY, "***MASKED***")
        # Also mask any partial key that might appear
        if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
            error_msg = error_msg.replace(OPENAI_API_KEY[:10], "***MASKED***")
        return f"[LLM error] {error_msg}"


def build_rag_prompt(question: str, contexts: List[Tuple[str, float, Dict]], conversation_context: str = "") -> str:
    context_texts: List[str] = []
    for i, (text, score, metadata) in enumerate(contexts, 1):
        page_info = f" (Page {metadata.get('page', '?')})" if metadata.get('page') else ""
        context_texts.append(f"[Context {i}{page_info}]: {text}")
    joined = "\n\n".join(context_texts)
    conversation_part = f"\n\nPrevious conversation:\n{conversation_context}\n" if conversation_context.strip() else ""
    prompt = f"""Answer this question using ONLY the document text below. Be direct and concise.

Document: {joined}

Question: {question}
Answer:"""
    return prompt


def build_summary_prompt(chunks: List[Tuple[str, float, Dict]], mode: str = "summary") -> str:
    context_texts: List[str] = []
    for i, (text, score, metadata) in enumerate(chunks, 1):
        page_info = f" (Page {metadata.get('page', '?')})" if metadata.get('page') else ""
        context_texts.append(f"[Context {i}{page_info}]: {text}")
    joined = "\n\n".join(context_texts)
    if mode == "bullets":
        instruction = "Produce 4-6 bullet points capturing the key facts."
    elif mode == "topics":
        instruction = "List 6-10 concise key topics/sections covered, as bullet points."
    else:
        instruction = "Produce a concise summary in 5-7 sentences."
    return f"""
Using ONLY the document text below, {instruction}

Document:
{joined}
"""


def _chat(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 300) -> str:
    if not OPENAI_API_KEY or not client:
        return ""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
    except Exception:
        return ""
    return ""


def generate_query_expansions(question: str, conversation_context: str = "") -> List[str]:
    """Use the LLM to create alternate phrasings and keyword queries to improve recall."""
    prompt = f"""
Rewrite the user's question into 3 alternative search queries and 5-10 compact keyword phrases.
Keep them short and content-bearing; avoid stopwords and pronouns. No numbering besides '- '.

Question: {question}
{('Conversation context: ' + conversation_context) if conversation_context else ''}
Return format:
- alt: <query 1>
- alt: <query 2>
- alt: <query 3>
- kw: <keywords, comma-separated>
"""
    out = _chat(prompt, system="You produce terse, query-like reformulations.", temperature=0.2, max_tokens=200)
    alts: List[str] = []
    kws: List[str] = []
    if out:
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith('- alt:'):
                alts.append(line.split(':', 1)[1].strip())
            elif line.lower().startswith('- kw:'):
                kw_str = line.split(':', 1)[1].strip()
                kws.extend([k.strip() for k in kw_str.split(',') if k.strip()])
    # Deduplicate and trim
    seen = set()
    alts = [q for q in alts if not (q in seen or seen.add(q))]
    seen = set()
    kws = [k for k in kws if not (k in seen or seen.add(k))]
    return alts + kws[:10]


def generate_hypothetical_answer(question: str, conversation_context: str = "") -> str:
    """HyDE: generate a concise hypothetical answer to embed for retrieval expansion."""
    prompt = f"""
Provide a concise, factual-style answer to the question as if the information were present in a document. 3-5 sentences max.
Avoid speculation markers; write it like a summary to be embedded.

Question: {question}
{('Conversation context: ' + conversation_context) if conversation_context else ''}
"""
    return _chat(prompt, system="Write a dry, factual paragraph.", temperature=0.2, max_tokens=180)


