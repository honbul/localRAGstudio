# Local RAG Studio

Local RAG Studio is a fully local, Open WebUI-style chat app with a built-in RAG studio. It lets you manage multiple named knowledge bases, download and reuse sentence-transformers embedding models, and chat using a locally authenticated Codex CLI or Gemini CLI.

## What you get

- Open WebUI-like chat UI with streaming responses and conversation history
- Knowledge base creation with chunking + retrieval settings
- Embedding model management (download from Hugging Face, cache locally)
- Multi-KB retrieval at chat time with citations
- Fully offline runtime (after models are downloaded)

## Requirements

- Python 3.10+
- Node.js 18+
- Codex CLI authenticated via OAuth (`codex login`)
- Gemini CLI authenticated (cached credentials)

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## How to use

1) **Embedding Models**: add a model like `sentence-transformers/all-MiniLM-L6-v2`.
2) **Create KB**: choose a local path or upload a folder, set chunking + top-k.
3) **Chat**: select one or more KBs to ground answers (or chat with none). Citations include relevance scores.
4) **Provider**: choose Codex or Gemini; the provider is saved per conversation.
5) **Dual mode**: enable Dual to chat with both providers side-by-side (separate histories).
6) **Knowledge Bases**: rename, rebuild, or add more documents anytime.
7) **Progress**: KB ingest and embedding downloads show live status updates.

## Local folder helper

Browsers cannot reveal absolute local paths. If you want a native folder picker that sends only the selected path to the backend, use the helper:

```bash
python helper/send_folder_path.py
```

It opens a native folder dialog and POSTs the absolute path to:

```
http://localhost:8000/api/folder-path
```

To change the endpoint:

```bash
FOLDER_HELPER_ENDPOINT=http://localhost:8000/api/folder-path python helper/send_folder_path.py
```

## Storage layout

```
data/app.db                 # chat history (SQLite)
data/embeddings/            # cached sentence-transformers models
data/kbs/<kb_name>/
  ├── metadata.json
  └── vectors/
      ├── index.faiss
      └── records.jsonl
```

## Configuration

- `RAG_DATA_DIR`: base data directory (default `./data`)
- `RAG_DB_PATH`: SQLite path (default `./data/app.db`)
- `RAG_KBS_DIR`: KB folder (default `./data/kbs`)
- `RAG_EMBEDDINGS_DIR`: embeddings cache (default `./data/embeddings`)
- `EMBEDDING_DEVICE`: `cpu` or `cuda` (default `cpu`)
- `CODEX_CLI_CMD`: Codex CLI command (default `codex exec --json --skip-git-repo-check -`)
- `CODEX_TIMEOUT`: Codex timeout seconds (default `180`)
- `GEMINI_CLI_CMD_JSON`: Gemini CLI JSON command (default `gemini -o json`)
- `GEMINI_CLI_CMD_STREAM`: Gemini CLI streaming JSON command (default `gemini -o stream-json`)
- `GEMINI_TIMEOUT`: Gemini timeout seconds (default `180`)

## WSL path helper

If you run the backend in WSL but browse on Windows, use the WSL path helper button to convert a Windows path like `C:\\Users\\you\\Docs` to `/mnt/c/Users/you/Docs` for the Source path field.

## Tips

- Dense docs: chunk size 300–600; narrative docs: 800–1200.
- Keep overlap at 10–20% of chunk size.
- Use a fast embedding model for quick iteration; switch to a quality model for final KBs.

## Notes

- `.doc` is not supported; convert to `.docx` or `.pdf`.
- If Codex reports session permission issues: `sudo chown -R $(whoami) ~/.codex`.
- Gemini CLI prints startup logs before JSON; the backend ignores those automatically.
