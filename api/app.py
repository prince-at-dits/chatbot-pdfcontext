from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>PDF RAG - Vercel Demo</title>
    <style>
        body { font-family: system-ui, Arial, sans-serif; margin: 24px; text-align: center; }
        .card { border: 1px solid #ddd; padding: 24px; border-radius: 8px; max-width: 600px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="card">
        <h1>PDF RAG Chatbot</h1>
        <p>⚠️ This is a demo version for Vercel deployment.</p>
        <p>Full functionality requires:</p>
        <ul style="text-align: left;">
            <li>Local Ollama server</li>
            <li>Persistent storage (Redis/Database)</li>
            <li>Dedicated server for file processing</li>
        </ul>
        <p>Consider deploying on Railway, Render, or DigitalOcean for full features.</p>
    </div>
</body>
</html>
    """

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Vercel deployment successful"})

if __name__ == "__main__":
    app.run()