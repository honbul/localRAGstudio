import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = os.environ.get("CODEX_BRIDGE_HOST", "0.0.0.0")
    port: int = int(os.environ.get("CODEX_BRIDGE_PORT", "8001"))
    codex_model: str = os.environ.get("CODEX_MODEL", "codex")

    mode: str = os.environ.get("CODEX_MODE", "exec")
    daemon_cmd: str = os.environ.get("CODEX_DAEMON_CMD", "codex daemon --stdio")
    daemon_mode: str = os.environ.get("CODEX_DAEMON_MODE", "stdio")
    daemon_socket: str = os.environ.get("CODEX_DAEMON_SOCKET", "/tmp/codex-daemon.sock")

    cli_cmd: str = os.environ.get(
        "CODEX_CLI_CMD",
        "codex exec --json --skip-git-repo-check -",
    )
    cli_input: str = os.environ.get("CODEX_CLI_INPUT", "stdin")
    timeout: int = int(os.environ.get("CODEX_TIMEOUT", "120"))


settings = Settings()
