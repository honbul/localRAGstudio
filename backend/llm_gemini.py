import json
import shlex
import subprocess
from typing import Dict, Generator, List

from .settings import settings


class GeminiError(RuntimeError):
    pass


def build_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.title()}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _extract_json_object(lines: List[str]) -> dict | None:
    for line in reversed(lines):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_gemini(messages: List[Dict[str, str]]) -> str:
    prompt = build_prompt(messages)
    cmd = shlex.split(settings.gemini_cmd_json)
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=settings.gemini_timeout,
        )
    except FileNotFoundError as exc:
        raise GeminiError("Gemini CLI not found") from exc

    stdout = (result.stdout or "").splitlines()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise GeminiError(stderr or "Gemini CLI failed")

    data = _extract_json_object(stdout)
    if not data:
        raise GeminiError("Gemini CLI returned no JSON payload")
    response = data.get("response")
    if not response:
        raise GeminiError("Gemini CLI response missing")
    return str(response)


def stream_gemini(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    prompt = build_prompt(messages)
    cmd = shlex.split(settings.gemini_cmd_stream)
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
        raise GeminiError("Gemini CLI not found") from exc

    if not process.stdin or not process.stdout:
        raise GeminiError("Gemini CLI stdin/stdout unavailable")

    process.stdin.write(prompt)
    process.stdin.flush()
    process.stdin.close()

    for line in process.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "message" and data.get("role") == "assistant":
            content = data.get("content")
            if content:
                yield str(content)

    process.wait(timeout=settings.gemini_timeout)
    if process.returncode != 0:
        err = (process.stderr.read() if process.stderr else "").strip()
        raise GeminiError(err or "Gemini CLI failed")
