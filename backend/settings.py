import os
from dataclasses import dataclass


@dataclass
class Settings:
    data_dir: str = os.environ.get("RAG_DATA_DIR", "./data")
    db_path: str = os.environ.get("RAG_DB_PATH", "./data/app.db")
    kbs_dir: str = os.environ.get("RAG_KBS_DIR", "./data/kbs")
    embeddings_dir: str = os.environ.get("RAG_EMBEDDINGS_DIR", "./data/embeddings")

    codex_cmd: str = os.environ.get(
        "CODEX_CLI_CMD",
        "codex exec --json --skip-git-repo-check -",
    )
    codex_timeout: int = int(os.environ.get("CODEX_TIMEOUT", "180"))

    gemini_cmd_json: str = os.environ.get("GEMINI_CLI_CMD_JSON", "gemini -o json")
    gemini_cmd_stream: str = os.environ.get("GEMINI_CLI_CMD_STREAM", "gemini -o stream-json")
    gemini_timeout: int = int(os.environ.get("GEMINI_TIMEOUT", "180"))


settings = Settings()
