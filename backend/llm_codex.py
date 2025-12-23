import json
import shlex
import subprocess
from typing import Dict, Generator, List

from .settings import settings


class CodexError(RuntimeError):
    pass


def build_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.title()}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _extract_delta(data: dict) -> str | None:
    if "delta" in data and isinstance(data["delta"], dict):
        return data["delta"].get("content")
    if "message" in data and isinstance(data["message"], dict):
        msg = data["message"]
        if msg.get("role") == "assistant":
            return msg.get("content")
    if data.get("type") == "item.completed":
        item = data.get("item", {})
        if isinstance(item, dict) and item.get("type") == "agent_message":
            return item.get("text")
    if "final_message" in data:
        return data.get("final_message")
    if "result" in data:
        return data.get("result")
    if "output" in data:
        return data.get("output")
    if "content" in data:
        return data.get("content")
    if "text" in data:
        return data.get("text")
    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return message.get("content")
    return None


def run_codex(messages: List[Dict[str, str]]) -> str:
    prompt = build_prompt(messages)
    cmd = shlex.split(settings.codex_cmd)
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=settings.codex_timeout,
        )
    except FileNotFoundError as exc:
        raise CodexError("Codex CLI not found") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise CodexError(stderr or stdout or "Codex CLI failed")

    if not stdout:
        return ""

    collected = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            collected.append(line)
            continue
        delta = _extract_delta(data)
        if delta:
            collected.append(str(delta))
    if collected:
        return "".join(collected).strip()
    return stdout


def stream_codex(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    prompt = build_prompt(messages)
    cmd = shlex.split(settings.codex_cmd)
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise CodexError("Codex CLI not found") from exc

    if not process.stdin or not process.stdout:
        raise CodexError("Codex CLI stdin/stdout unavailable")

    process.stdin.write(prompt)
    process.stdin.flush()
    process.stdin.close()

    fallback_buffer = []
    emitted = False
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            fallback_buffer.append(line)
            continue
        delta = _extract_delta(data)
        if delta:
            emitted = True
            yield str(delta)

    process.wait(timeout=settings.codex_timeout)
    if process.returncode != 0:
        err = (process.stderr.read() if process.stderr else "").strip()
        raise CodexError(err or "Codex CLI failed")

    if fallback_buffer and not emitted:
        yield "\n".join(fallback_buffer)
