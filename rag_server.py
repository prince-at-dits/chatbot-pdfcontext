from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI


app = Flask(__name__)
CORS(app, supports_credentials=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@app.route("/", methods=["GET"])
def root():
	return (
		"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Minimal Chat</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
      .card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; max-width: 800px; }
      textarea { width: 100%; height: 120px; }
      pre { background: #f7f7f7; padding: 12px; border-radius: 6px; white-space: pre-wrap; }
      button { padding: 8px 12px; }
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h2>Simple OpenAI Chat</h2>
      <textarea id=\"q\" placeholder=\"Say hi or ask anything...\"></textarea>
      <button onclick=\"ask()\">Send</button>
      <pre id=\"answer\"></pre>
    </div>
    <script>
      async function ask() {
        const q = document.getElementById('q').value.trim();
        if (!q) return;
        const ans = document.getElementById('answer');
        ans.textContent = 'Thinking...';
        try {
          const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: q }) });
          const j = await res.json();
          ans.textContent = j.answer || JSON.stringify(j, null, 2);
        } catch (e) {
          ans.textContent = 'Error: ' + (e && e.message ? e.message : 'request failed');
        }
      }
    </script>
  </body>
</html>
		"""
	)


@app.route("/chat", methods=["POST"])
def chat():
	if not client:
		return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500
	try:
		payload = request.get_json(force=True)
	except Exception:
		payload = {}
	message = (payload.get("message") or "").strip()
	if not message:
		return jsonify({"ok": False, "error": "message is required"}), 400

	try:
		resp = client.chat.completions.create(
			model=OPENAI_MODEL,
			messages=[
				{"role": "system", "content": "You are a helpful, concise assistant."},
				{"role": "user", "content": message}
			],
			temperature=0.7,
			max_tokens=500,
			timeout=60
		)
		content = (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
		if not content:
			content = "[No response]"
		return jsonify({"ok": True, "answer": content})
	except Exception as e:
		return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
	host = os.getenv("HOST", "0.0.0.0")
	port = int(os.getenv("PORT", "8081"))
	debug = os.getenv("DEBUG", "false").lower() == "true"
	app.run(host=host, port=port, debug=debug)


