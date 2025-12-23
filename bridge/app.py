import time
import uuid
import json
from typing import Dict, List, Generator, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .codex_cli import CodexCLIClient, CodexCLIConfig, CodexCLIError
from .settings import settings


app = FastAPI()

codex_client = CodexCLIClient(
    CodexCLIConfig(
        mode=settings.mode,
        model=settings.codex_model,
        timeout=settings.timeout,
        daemon_cmd=settings.daemon_cmd,
        daemon_mode=settings.daemon_mode,
        daemon_socket=settings.daemon_socket,
        cli_cmd=settings.cli_cmd,
        cli_input=settings.cli_input,
    )
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: List[ChatMessage]
    stream: bool | None = False


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.codex_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "codex-cli",
            }
        ],
    }


def _build_response(content: str, model: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _stream_chunks(chunks: Generator[str, None, None], model: str):
    for chunk in chunks:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    model = req.model or settings.codex_model
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages are required")

    messages = [m.model_dump() for m in req.messages]
    try:
        if req.stream:
            chunks = list(codex_client.stream_chat(messages))
            return StreamingResponse(
                _stream_chunks(chunks, model),
                media_type="text/event-stream",
            )
        content = codex_client.chat(messages)
    except CodexCLIError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Codex bridge error: {exc}") from exc

    return _build_response(content, model)


@app.get("/")
def root():
    return {"status": "ok", "service": "codex-bridge"}
