import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8081")

@app.route("/", methods=["GET"])
def index():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PDF RAG Chatbot</title>
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
    <div class="card">
      <h2>Upload PDFs</h2>
      <input id="pdfs" type="file" multiple accept="application/pdf" />
      <button onclick="upload()">Ingest</button>
      <pre id="ingestStatus"></pre>
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
    <script>
      const BACKEND_URL = window.location.hostname === 'localhost' ? 'http://localhost:8081' : '';
      
      async function upload() {
        const f = document.getElementById('pdfs');
        const fd = new FormData();
        for (const file of f.files) fd.append('files', file);
        const res = await fetch(BACKEND_URL + '/upload', { method: 'POST', body: fd });
        const j = await res.json();
        document.getElementById('ingestStatus').textContent = JSON.stringify(j, null, 2);
      }
      
      async function loadConversation() {
        try {
          const res = await fetch(BACKEND_URL + '/conversation');
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
          const res = await fetch(BACKEND_URL + '/ask', { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({ question: q }) 
          });
          const j = await res.json();
          
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
      
      window.onload = () => loadConversation();
    </script>
  </body>
</html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)