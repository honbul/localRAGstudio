import os
import json
import time
from typing import List, Dict, Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .settings import settings
from .db import init_db
from .history import (
    list_conversations,
    create_conversation,
    delete_conversation,
    get_messages,
    add_message,
    ensure_title,
    set_provider,
)
from .embeddings import list_models, register_model, delete_model
from .kb_manager import list_kbs, ingest, ingest_into, delete_kb, rename_kb, rebuild, get_kb
from .kb_manager import save_kb
from .retrieval import retrieve, build_context
from .llm_codex import run_codex, stream_codex, CodexError
from .llm_gemini import run_gemini, stream_gemini, GeminiError


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


class ConversationCreate(BaseModel):
    title: str | None = None
    provider: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    kb_names: List[str] | None = None
    top_k: int | None = None
    stream: bool | None = False
    rag_mode: str | None = "hybrid"
    provider: str | None = "codex"


class KBCreate(BaseModel):
    name: str
    source_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


class KBIngest(BaseModel):
    source_path: str


class KBRename(BaseModel):
    name: str


class EmbeddingCreate(BaseModel):
    model_id: str
    tag: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/conversations")
def api_list_conversations():
    return {"conversations": list_conversations()}


@app.post("/api/conversations")
def api_create_conversation(req: ConversationCreate):
    return create_conversation(req.title, req.provider)


@app.delete("/api/conversations/{convo_id}")
def api_delete_conversation(convo_id: str):
    delete_conversation(convo_id)
    return {"status": "deleted"}


@app.get("/api/conversations/{convo_id}/messages")
def api_get_messages(convo_id: str):
    return {"messages": get_messages(convo_id)}


@app.get("/api/embeddings/models")
def api_list_models():
    models = list_models()
    return {"models": [model.__dict__ for model in models]}


@app.post("/api/embeddings/models")
def api_add_model(req: EmbeddingCreate):
    info = register_model(req.model_id, req.tag)
    return info.__dict__


@app.delete("/api/embeddings/models/{model_id:path}")
def api_delete_model(model_id: str):
    delete_model(model_id)
    return {"status": "deleted"}


@app.get("/api/kbs")
def api_list_kbs():
    return {"kbs": list_kbs()}


@app.post("/api/kbs")
def api_create_kb(req: KBCreate):
    if not os.path.exists(req.source_path):
        raise HTTPException(status_code=400, detail="source path does not exist")
    try:
        result = ingest(
            req.name,
            req.source_path,
            req.embedding_model,
            req.chunk_size,
            req.chunk_overlap,
            req.top_k,
        )
        return result
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/kbs/{kb_name}/ingest")
def api_ingest_kb(kb_name: str, req: KBIngest):
    if not os.path.exists(req.source_path):
        raise HTTPException(status_code=400, detail="source path does not exist")
    meta = get_kb(kb_name)
    try:
        result = ingest_into(
            kb_name,
            req.source_path,
            meta["embedding_model"],
            meta["chunk_size"],
            meta["chunk_overlap"],
            meta["top_k"],
        )
        meta.update(result["meta"])
        save_kb(meta)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/kbs/{kb_name}/rebuild")
def api_rebuild_kb(kb_name: str, req: KBIngest):
    if not os.path.exists(req.source_path):
        raise HTTPException(status_code=400, detail="source path does not exist")
    return rebuild(kb_name, req.source_path)


@app.patch("/api/kbs/{kb_name}")
def api_rename_kb(kb_name: str, req: KBRename):
    return rename_kb(kb_name, req.name)


@app.delete("/api/kbs/{kb_name}")
def api_delete_kb(kb_name: str):
    delete_kb(kb_name)
    return {"status": "deleted"}


def _build_chat_messages(
    history: List[Dict[str, Any]],
    context: str | None,
    rag_mode: str,
) -> List[Dict[str, str]]:
    messages = []
    if context:
        if rag_mode == "rag_only":
            system_intro = (
                "Answer using only the provided context. "
                "If the answer is not in the context, say you do not have enough information."
            )
        else:
            system_intro = (
                "Use the provided context when it is relevant. "
                "If the question is general and does not require the context, answer normally. "
                "Only say you do not have enough information when the answer depends on missing context."
            )
        messages.append({
            "role": "system",
            "content": f"{system_intro}\n\nContext:\n{context}",
        })
    for msg in history:
        if msg["role"] in {"user", "assistant", "system"}:
            messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def _stream_response(
    chunks: Generator[str, None, None],
    convo_id: str,
    sources: list | None,
):
    collected = []
    try:
        for chunk in chunks:
            collected.append(chunk)
            payload = {"type": "delta", "content": chunk}
            yield f"data: {json.dumps(payload)}\n\n"
    except CodexError as exc:
        error_payload = {"type": "error", "message": str(exc)}
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as exc:
        error_payload = {"type": "error", "message": f"Stream error: {exc}"}
        yield f"data: {json.dumps(error_payload)}\n\n"
    answer = "".join(collected).strip()
    if answer:
        add_message(convo_id, "assistant", answer, sources)
    final = {"type": "done", "conversation_id": convo_id, "sources": sources}
    yield f"data: {json.dumps(final)}\n\n"


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    convo_id = req.conversation_id
    if not convo_id:
        convo = create_conversation(provider=req.provider)
        convo_id = convo["id"]

    add_message(convo_id, "user", req.message)
    ensure_title(convo_id, req.message)
    if req.provider:
        set_provider(convo_id, req.provider)

    history = get_messages(convo_id)
    kb_names = req.kb_names or []
    top_k = req.top_k or 5
    context = None
    sources = None
    if kb_names:
        hits = retrieve(kb_names, req.message, top_k)
        context = build_context(hits)
        sources = hits

    prompt_messages = _build_chat_messages(history, context, req.rag_mode or "hybrid")

    provider = (req.provider or "codex").lower()
    try:
        if req.stream:
            if provider == "gemini":
                chunks = stream_gemini(prompt_messages)
            else:
                chunks = stream_codex(prompt_messages)
            return StreamingResponse(
                _stream_response(chunks, convo_id, sources),
                media_type="text/event-stream",
            )
        if provider == "gemini":
            answer = run_gemini(prompt_messages)
        else:
            answer = run_codex(prompt_messages)
    except (CodexError, GeminiError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    add_message(convo_id, "assistant", answer, sources)

    return {
        "conversation_id": convo_id,
        "answer": answer,
        "sources": sources,
    }
