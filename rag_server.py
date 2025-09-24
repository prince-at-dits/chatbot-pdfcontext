from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

import os
import json
import uuid
import re
from typing import List, Dict, Tuple

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from session_store import SessionIndex
from lc_pipeline import LCPipeline


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "44v1usagCjTZLFBXzgxUerRBAxBHk0bL")
CORS(app, 
     supports_credentials=True,
     origins=['http://localhost:3000', 'https://pdf-rag-frontend-sandy.vercel.app', 'https://crud-backend-uwh6.onrender.com'],
     allow_headers=['Content-Type', 'X-Session-ID'],
     methods=['GET', 'POST', 'OPTIONS'])


SESSION_STORES: Dict[str, SessionIndex] = {}


def get_or_create_session_store() -> SessionIndex:
    # Try to get session ID from header first (for cross-origin), then from session
    sid = request.headers.get('X-Session-ID') or session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    store = SESSION_STORES.get(sid)
    if store is None:
        store = SessionIndex()
        SESSION_STORES[sid] = store
    return store, sid


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
    store, sid = get_or_create_session_store()
    if "files" not in request.files:
        return jsonify({"ok": False, "error": "No files part"}), 400
    files = request.files.getlist("files")
    for file in files:
        if not file or file.filename == "":
            continue
        filename = secure_filename(file.filename)
        try:
            data = file.read()
            # File size validation: limit 5 MB
            if len(data) > 5 * 1024 * 1024:
                return jsonify({"ok": False, "file": filename, "error": "File too large (>5MB)"}), 400
            
            # Build LangChain pipeline per session using helper
            if store.lc_pipeline is None:
                store.lc_pipeline = LCPipeline()
            store.lc_pipeline.build_from_pdf_bytes(data, filename)
            
            # Track filename
            store.uploaded_filenames[filename] = None
            
        except Exception as e:
            return jsonify({"ok": False, "file": filename, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "session_id": sid,
        "pages_by_file": store.get_pages_by_file(),
        "total_pages": store.get_total_pages()
    })


@app.route("/ask", methods=["POST"])
def ask():
    store, sid = get_or_create_session_store()
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400
    
    store.add_to_conversation("user", question)
    
    # Check if this is a general greeting/conversation
    general_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'thanks', 'thank you', 'bye', 'goodbye']
    lower_q = question.lower()
    is_general_greeting = any(keyword in lower_q for keyword in general_keywords)

    # If no LangChain pipeline exists, handle greetings or ask to upload
    if store.lc_pipeline is None or store.lc_pipeline.qa_chain is None:
        if is_general_greeting:
            answer = "Hello! Please upload a PDF to get started."
        else:
            answer = "Please upload a PDF first. I can only answer greetings until a document is provided."
        store.add_to_conversation("assistant", answer)
        return jsonify({
            "ok": True,
            "answer": answer,
            "sources": [],
            "is_general": True
        })

    # PDFs exist; if greeting, acknowledge and steer towards document questions
    if is_general_greeting:
        filenames = store.get_uploaded_filenames()
        file_hint = f" {', '.join(filenames)}" if filenames else ""
        answer = (
            "Hello! I have your uploaded PDF(s)"
            + (f": {file_hint}. " if file_hint else ". ")
            + "Ask me questions about their content."
        )
        store.add_to_conversation("assistant", answer)
        return jsonify({
            "ok": True,
            "answer": answer,
            "sources": [],
            "is_general": True
        })

    # Handle page count requests
    if any(k in lower_q for k in ["how many pages", "pages in", "page count", "total pages"]):
        pages = store.get_total_pages() or 0
        filenames = store.get_uploaded_filenames()
        if pages > 0:
            answer = f"This session has {pages} page(s) across: {', '.join(filenames)}."
        else:
            answer = "I couldn't determine the page count from the uploaded PDF."
        store.add_to_conversation("assistant", answer)
        return jsonify({"ok": True, "answer": answer, "sources": [], "is_general": False})

    # Handle summary requests
    if any(k in lower_q for k in ["summary", "summarize", "overview", "whole", "entire", "full", "complete"]):
        try:
            answer = store.lc_pipeline.summarize(n_sentences=12)
        except Exception as e:
            answer = f"I couldn't summarize due to an internal error: {e}"
        store.add_to_conversation("assistant", answer)
        return jsonify({"ok": True, "answer": answer, "sources": [], "is_general": False})

    # General QA via LangChain pipeline
    try:
        answer = store.lc_pipeline.ask(question) or "I don't have enough information about that."
    except Exception as e:
        answer = f"I couldn't answer due to an internal error: {e}"
    store.add_to_conversation("assistant", answer)
    return jsonify({"ok": True, "answer": answer, "sources": [], "is_general": False})


@app.route("/debug-session", methods=["GET"])
def debug_session():
    """Debug endpoint to check session state"""
    store, sid = get_or_create_session_store()
    return jsonify({
        "ok": True,
        "session_id": sid,
        "conversation_length": len(store.conversation_history),
        "has_pipeline": store.lc_pipeline is not None,
        "pipeline_metadata": store.lc_pipeline.metadata if store.lc_pipeline else None
    })


@app.route("/conversation", methods=["GET"])
def get_conversation():
    """Get conversation history for current session"""
    store, sid = get_or_create_session_store()
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


