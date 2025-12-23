import json
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Dict, Generator, List, Optional

from .prompt import build_prompt


class CodexCLIError(RuntimeError):
    pass


@dataclass
class CodexCLIConfig:
    mode: str
    model: str
    timeout: int
    daemon_cmd: str
    daemon_mode: str
    daemon_socket: str
    cli_cmd: str
    cli_input: str


class CodexCLIClient:
    def __init__(self, config: CodexCLIConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def _start_daemon(self) -> None:
        cmd = shlex.split(self.config.daemon_cmd)
        if not cmd:
            raise CodexCLIError("CODEX_DAEMON_CMD is empty")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise CodexCLIError("Codex CLI not found for daemon command") from exc

    def _ensure_stdio_daemon(self) -> subprocess.Popen[str]:
        if self._process and self._process.poll() is None:
            return self._process
        self._start_daemon()
        if not self._process:
            raise CodexCLIError("Failed to start Codex daemon")
        return self._process

    def _ensure_socket_daemon(self) -> None:
        if os.path.exists(self.config.daemon_socket):
            return
        self._start_daemon()
        timeout_at = time.time() + self.config.timeout
        while time.time() < timeout_at:
            if os.path.exists(self.config.daemon_socket):
                return
            time.sleep(0.2)
        raise CodexCLIError("Timed out waiting for Codex socket")

    def _readline_with_timeout(self, stream, timeout: int) -> str:
        queue: Queue[str] = Queue()

        def _reader() -> None:
            line = stream.readline()
            queue.put(line)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        try:
            return queue.get(timeout=timeout)
        except Empty:
            raise CodexCLIError("Timed out waiting for Codex response")

    def _extract_content(self, raw: str) -> str | None:
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if "error" in data:
            raise CodexCLIError(str(data["error"]))
        if "content" in data:
            return str(data["content"])
        if "response" in data:
            return str(data["response"])
        if "text" in data:
            return str(data["text"])
        if "choices" in data:
            choices = data.get("choices") or []
            if choices:
                choice = choices[0]
                message = choice.get("message") or {}
                content = message.get("content")
                if content:
                    return str(content)
        return ""

    def _daemon_request(self, payload: Dict[str, object]) -> str:
        if self.config.daemon_mode == "socket":
            return self._daemon_socket(payload)
        return self._daemon_stdio(payload)

    def _daemon_stdio(self, payload: Dict[str, object]) -> str:
        process = self._ensure_stdio_daemon()
        if not process.stdin or not process.stdout:
            raise CodexCLIError("Codex daemon stdio not available")
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

        deadline = time.time() + self.config.timeout
        last_line = ""
        while time.time() < deadline:
            remaining = max(0.1, min(2.0, deadline - time.time()))
            line = self._readline_with_timeout(process.stdout, remaining)
            if not line:
                continue
            last_line = line.strip()
            content = self._extract_content(last_line)
            if content is not None:
                return content
        if last_line:
            return last_line
        raise CodexCLIError("Empty response from Codex daemon")

    def _daemon_socket(self, payload: Dict[str, object]) -> str:
        self._ensure_socket_daemon()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.config.timeout)
            client.connect(self.config.daemon_socket)
            client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            chunks = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8")
        if not raw:
            raise CodexCLIError("Empty response from Codex daemon")
        content = self._extract_content(raw)
        if content is not None:
            return content
        return raw.strip()

    def _exec_request(self, prompt: str) -> str:
        cmd = shlex.split(self.config.cli_cmd)
        if not cmd:
            raise CodexCLIError("CODEX_CLI_CMD is empty")
        if self.config.cli_input == "arg":
            cmd.append(prompt)
            input_data = None
        else:
            input_data = prompt

        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except FileNotFoundError as exc:
            raise CodexCLIError("Codex CLI not found for exec command") from exc
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            detail = stderr or stdout or "Codex CLI failed"
            raise CodexCLIError(detail)
        if not stdout:
            return ""
        content = self._extract_from_jsonl(stdout)
        return content if content is not None else stdout

    def _extract_from_jsonl(self, output: str) -> str | None:
        chunks: List[str] = []
        last_message: Optional[str] = None
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                if "error" in data:
                    raise CodexCLIError(str(data["error"]))

                if "message" in data and isinstance(data["message"], dict):
                    msg = data["message"]
                    if msg.get("role") == "assistant" and msg.get("content"):
                        last_message = str(msg["content"])
                        continue

                if "delta" in data and isinstance(data["delta"], dict):
                    delta = data["delta"].get("content")
                    if delta:
                        chunks.append(str(delta))
                        continue

                for key in ("content", "text", "response"):
                    if key in data and data[key]:
                        last_message = str(data[key])
                        break

                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    message = choice.get("message") or {}
                    content = message.get("content")
                    if content:
                        last_message = str(content)
                        continue

        if chunks:
            return "".join(chunks).strip()
        if last_message is not None:
            return last_message.strip()
        return None

    def _run_prompt(self, prompt: str) -> str:
        if self.config.mode == "daemon":
            payload = {
                "type": "chat",
                "model": self.config.model,
                "prompt": prompt,
            }
            return self._daemon_request(payload)
        return self._exec_request(prompt)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = build_prompt(messages)
        with self._lock:
            return self._run_prompt(prompt)

    def stream_chat(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        response = self.chat(messages)
        if not response:
            return
        chunk = []
        for token in response.split():
            chunk.append(token)
            if len(chunk) >= 12:
                yield " ".join(chunk) + " "
                chunk = []
        if chunk:
            yield " ".join(chunk)
