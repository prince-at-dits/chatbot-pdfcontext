## Chatbot PDF Context (RAG Server)

A lightweight Flask app that lets you upload PDFs and ask questions answered with Retrieval Augmented Generation (RAG). Embeddings are created with `sentence-transformers`, indexed in-memory using `faiss`, and answers are generated via an Ollama model.

### Features
- Upload multiple PDFs; chunks are embedded and stored per-browser session
- Ask questions; top matches are used to build a RAG prompt
- General conversation fallback if no PDFs uploaded
- Debug endpoints for search thresholds and test prompts

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com) running locally with an available model (default: `llama3.2`)

### Quickstart
1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start Ollama (example)
```bash
# Install model if needed
ollama pull llama3.2
# Ensure the Ollama server is running (usually auto-starts)
```

4. Configure environment (optional)
Create a `.env` file in the project root to override defaults:
```bash
cp .env.example .env  # if you create an example first, otherwise create .env directly
```
Available variables:
- `FLASK_SECRET_KEY` – Flask session secret
- `HOST` – Bind host (default `0.0.0.0`)
- `PORT` – Port (default `8081`)
- `DEBUG` – `true`/`false` (default `false`)
- `OLLAMA_URL` – e.g., `http://localhost:11434`
- `OLLAMA_MODEL` – e.g., `llama3.2`
- `EMBED_MODEL` – sentence-transformers model (default `sentence-transformers/all-MiniLM-L6-v2`)

5. Run the server
```bash
python rag_server.py
```
Open `http://localhost:8081` in your browser.

### Endpoints
- `GET /` – Minimal UI for upload and Q&A
- `POST /upload` – Form-data `files` (PDFs), ingests chunks and embeddings
- `POST /ask` – JSON `{ "question": "..." }`, returns answer and sources
- `POST /debug-search` – JSON `{ "question": "..." }`, returns threshold diagnostics
- `GET /test-prompts` – Returns sample prompts grouped by category

### Notes
- Embeddings and conversation history are stored per-session in memory and are not persisted. Restarting the server clears state.
- `faiss-cpu` is used; ensure your environment supports it. For some platforms you may need to adjust versions.
- PDF text extraction uses `PyPDF2`; quality depends on the source PDF.

### Development
- Format/linters are not enforced; feel free to add `ruff`/`black`.
- Contributions: open PRs/issues.
