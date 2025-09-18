from dotenv import load_dotenv
import os
import json
import uuid
from typing import List, Dict, Tuple

from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename

from embeddings import embed_texts, EMBEDDING_DIMENSION
from session_store import SessionIndex
from text_utils import pdf_to_text_chunks
from llm_client import generate_with_ollama, build_rag_prompt


load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "44v1usagCjTZLFBXzgxUerRBAxBHk0bL")


SESSION_STORES: Dict[str, SessionIndex] = {}


from embeddings import embed_texts


def get_or_create_session_store() -> SessionIndex:
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    store = SESSION_STORES.get(sid)
    if store is None:
        store = SessionIndex(EMBEDDING_DIMENSION)
        SESSION_STORES[sid] = store
    return store
from llm_client import generate_with_ollama, build_rag_prompt


@app.route("/", methods=["GET"])
def index():
    get_or_create_session_store()
    return (
        """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>PDF RAG (Per-Browser Session)</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
      .card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; max-width: 900px; }
      input[type=file] { margin-top: 8px; }
      textarea { width: 100%; height: 120px; }
      pre { background: #f7f7f7; padding: 12px; border-radius: 6px; white-space: pre-wrap; }
      button { padding: 8px 12px; }
      .msg { margin: 8px 0; padding: 8px; border-radius: 6px; }
      .user { background: #e3f2fd; text-align: right; }
      .assistant { background: #f1f8e9; }
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h2>Upload PDFs</h2>
      <input id=\"pdfs\" type=\"file\" multiple accept=\"application/pdf\" />
      <button onclick=\"upload()\">Ingest</button>
      <pre id=\"ingestStatus\"></pre>
    </div>
    <br/>
    <div class="card">
      <h2>Conversation</h2>
      <div id="conversation" style="max-height:400px; overflow-y:auto; border:1px solid #eee; padding:12px; margin-bottom:12px; background:#fafafa;"></div>
      <textarea id="q" placeholder="Type your question about the uploaded PDFs..."></textarea>
      <button id="askBtn" onclick="ask()">Ask</button>
      <span id="askLoading" style="display:none; margin-left:8px; color:#555;">Searching…</span>
      <div id="answerInfo" style="margin-top:8px; font-size:12px; color:#666;"></div>
    </div>
    <br/>
    <div class="card">
      <h2>Test Prompts</h2>
      <button onclick="loadTestPrompts()">Load Test Prompts</button>
      <button onclick="debugSearch()" style="margin-left:8px;">Debug Search</button>
      <div id="testPrompts" style="margin-top:8px;"></div>
      <div id="debugResults" style="margin-top:8px;"></div>
    </div>
    <script>
      async function upload() {
        const f = document.getElementById('pdfs');
        const fd = new FormData();
        for (const file of f.files) fd.append('files', file);
        const res = await fetch('/upload', { method: 'POST', body: fd });
        const j = await res.json();
        document.getElementById('ingestStatus').textContent = JSON.stringify(j, null, 2);
      }
      async function loadConversation() {
        try {
          const res = await fetch('/conversation');
          const j = await res.json();
          const conv = document.getElementById('conversation');
          if (j.conversation && j.conversation.length > 0) {
            conv.innerHTML = j.conversation.map(msg => 
              `<div class="msg ${msg.role}">${msg.content}</div>`
            ).join('');
            conv.scrollTop = conv.scrollHeight;
          } else {
            conv.innerHTML = '<div style="color:#999; text-align:center;">No conversation yet. Ask a question to start!</div>';
          }
        } catch (e) {
          console.error('Failed to load conversation:', e);
        }
      }
      
      async function ask() {
        const q = document.getElementById('q').value.trim();
        if (!q) return;
        
        const btn = document.getElementById('askBtn');
        const loader = document.getElementById('askLoading');
        const info = document.getElementById('answerInfo');
        btn.disabled = true;
        loader.style.display = 'inline';
        info.textContent = '';
        
        try {
          const res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) });
          const j = await res.json();
          
          // Show additional info
          let infoText = '';
          if (j.is_general) {
            infoText = 'General conversation (no document context)';
          } else if (j.sources && j.sources.length > 0) {
            infoText = `Found ${j.sources.length} relevant sources`;
            if (j.sources[0].score) {
              infoText += ` (best match: ${(j.sources[0].score * 100).toFixed(1)}%)`;
            }
          } else {
            infoText = 'No relevant sources found in documents';
          }
          info.textContent = infoText;
          
          document.getElementById('q').value = '';
          loadConversation();
        } catch (e) {
          info.textContent = 'Error: ' + (e && e.message ? e.message : 'request failed');
        } finally {
          loader.style.display = 'none';
          btn.disabled = false;
        }
      }
      
      async function loadTestPrompts() {
        try {
          const res = await fetch('/test-prompts');
          const prompts = await res.json();
          const container = document.getElementById('testPrompts');
          let html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">';
          
          for (const [category, questions] of Object.entries(prompts)) {
            html += `<div><h4>${category.replace(/_/g, ' ').toUpperCase()}</h4><ul>`;
            questions.forEach(q => {
              html += `<li><button onclick="document.getElementById('q').value='${q.replace(/'/g, '\\'')}'; ask();" style="text-align:left; background:none; border:none; color:#0066cc; cursor:pointer; padding:2px;">${q}</button></li>`;
            });
            html += '</ul></div>';
          }
          html += '</div>';
          container.innerHTML = html;
        } catch (e) {
          document.getElementById('testPrompts').innerHTML = 'Error loading test prompts';
        }
      }
      
      async function debugSearch() {
        const q = document.getElementById('q').value;
        if (!q.trim()) {
          alert('Please enter a question first');
          return;
        }
        try {
          const res = await fetch('/debug-search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) });
          const j = await res.json();
          document.getElementById('debugResults').innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
        } catch (e) {
          document.getElementById('debugResults').innerHTML = 'Debug error: ' + e.message;
        }
      }
      
      // Load conversation on page load
      window.onload = () => loadConversation();
    </script>
  </body>
</html>
        """
    )


@app.route("/upload", methods=["POST"])
def upload():
    store = get_or_create_session_store()
    if "files" not in request.files:
        return jsonify({"ok": False, "error": "No files part"}), 400
    files = request.files.getlist("files")
    total_chunks = 0
    for file in files:
        if not file or file.filename == "":
            continue
        filename = secure_filename(file.filename)
        try:
            data = file.read()
            chunks_with_metadata = pdf_to_text_chunks(data)
            if not chunks_with_metadata:
                continue
            
            chunks = [chunk for chunk, _ in chunks_with_metadata]
            metadata = [meta for _, meta in chunks_with_metadata]
            embeddings = embed_texts(chunks)
            store.add(embeddings, chunks, metadata)
            total_chunks += len(chunks)
        except Exception as e:
            return jsonify({"ok": False, "file": filename, "error": str(e)}), 500
    return jsonify({"ok": True, "chunks_added": total_chunks, "index_size": store.index.ntotal})


@app.route("/ask", methods=["POST"])
def ask():
    store = get_or_create_session_store()
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400
    
    store.add_to_conversation("user", question)
    
    if store.index.ntotal == 0:
        general_prompt = f"""You are a helpful AI assistant. The user is asking: {question}

Please provide a helpful response. If this seems like a question that would benefit from document context, suggest that they upload relevant PDFs first.

Previous conversation:
{store.get_conversation_context()}

Answer:"""
        answer = generate_with_ollama(
            system_prompt="You are a helpful AI assistant. Be conversational and helpful.",
            user_prompt=general_prompt
        )
        store.add_to_conversation("assistant", answer)
        return jsonify({
            "ok": True,
            "answer": answer,
            "sources": [],
            "is_general": True
        })
    
    query_emb = embed_texts([question])
    contexts = store.search(query_emb, top_k=6, min_score=0.1)
    
    conversation_context = store.get_conversation_context()
    
    if not contexts:
        broader_contexts = store.search(query_emb, top_k=10, min_score=0.05)
        if broader_contexts:
            contexts = broader_contexts[:3] 
            answer = "I found some potentially relevant information, though the match isn't perfect:\n\n"
        else:
            answer = f"I don't have enough relevant information in the uploaded documents to answer this question. The document contains {store.index.ntotal} text chunks. Could you rephrase your question or upload more relevant documents?"
            store.add_to_conversation("assistant", answer)
            return jsonify({
                "ok": True,
                "answer": answer,
                "sources": [],
                "is_general": False,
                "debug_info": f"Total chunks: {store.index.ntotal}, Search threshold: 0.05"
            })
    
    prompt = build_rag_prompt(question, contexts, conversation_context)
    
    answer = generate_with_ollama(
        system_prompt="You are a helpful AI assistant with access to document context. Answer accurately based on the provided information.",
        user_prompt=prompt
    )
    
    store.add_to_conversation("assistant", answer)
    
    return jsonify({
        "ok": True,
        "answer": answer,
        "sources": [{"text": c, "score": s, "metadata": m} for c, s, m in contexts],
        "is_general": False
    })


@app.route("/debug-search", methods=["POST"])
def debug_search():
    """Debug endpoint to see what's happening with search"""
    store = get_or_create_session_store()
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400
    
    query_emb = embed_texts([question])
    
    results = {}
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        contexts = store.search(query_emb, top_k=10, min_score=threshold)
        results[f"threshold_{threshold}"] = {
            "count": len(contexts),
            "scores": [float(score) for _, score, _ in contexts],
            "samples": [text[:100] + "..." for text, _, _ in contexts[:3]]
        }
    
    return jsonify({
        "ok": True,
        "question": question,
        "total_chunks": store.index.ntotal,
        "search_results": results
    })


@app.route("/conversation", methods=["GET"])
def get_conversation():
    """Get conversation history for current session"""
    store = get_or_create_session_store()
    with store.lock:
        return jsonify({
            "ok": True,
            "conversation": store.conversation_history
        })


@app.route("/test-prompts", methods=["GET"])
def test_prompts():
    """Return test prompts for evaluating accuracy"""
    return jsonify({
        "basic_conversation": [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "What can you help me with?",
            "Thank you for your help"
        ],
        "document_questions": [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the main conclusions?",
            "Who is the author?",
            "What methodology was used?",
            "What are the limitations mentioned?",
            "What recommendations are given?",
            "What data or statistics are presented?",
            "What are the main arguments?",
            "What future work is suggested?"
        ],
        "specific_fact_questions": [
            "What specific numbers or percentages are mentioned?",
            "What dates or time periods are referenced?",
            "What names or organizations are mentioned?",
            "What technical terms or concepts are defined?",
            "What examples or case studies are provided?"
        ],
        "comparison_questions": [
            "How does X compare to Y?",
            "What are the differences between A and B?",
            "What are the similarities mentioned?",
            "Which approach is better according to the document?"
        ],
        "inference_questions": [
            "What can we infer from this information?",
            "What are the implications of these findings?",
            "What would happen if...?",
            "Why do you think the author suggests this?"
        ]
    })


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)


