import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Callable

import numpy as np

from .chunking import chunk_text
from .documents import iter_paths, load_document, SUPPORTED_EXTENSIONS
from .embeddings import get_embedder
from .vector_store import FaissStore, VectorRecord
from .settings import settings


@dataclass
class KBMetadata:
    name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    created_at: float
    updated_at: float
    document_count: int
    chunk_count: int


def _kb_dir(name: str) -> str:
    safe = name.strip().replace(" ", "_")
    return os.path.join(settings.kbs_dir, safe)


def _meta_path(kb_dir: str) -> str:
    return os.path.join(kb_dir, "metadata.json")


def list_kbs() -> List[Dict[str, Any]]:
    if not os.path.exists(settings.kbs_dir):
        return []
    items = []
    for name in os.listdir(settings.kbs_dir):
        kb_dir = os.path.join(settings.kbs_dir, name)
        if not os.path.isdir(kb_dir):
            continue
        meta_path = _meta_path(kb_dir)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return sorted(items, key=lambda item: item.get("updated_at", 0), reverse=True)


def get_kb(name: str) -> Dict[str, Any]:
    kb_dir = _kb_dir(name)
    meta_path = _meta_path(kb_dir)
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Knowledge base not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_kb(meta: Dict[str, Any]) -> None:
    kb_dir = _kb_dir(meta["name"])
    os.makedirs(kb_dir, exist_ok=True)
    with open(_meta_path(kb_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def delete_kb(name: str) -> None:
    kb_dir = _kb_dir(name)
    if not os.path.exists(kb_dir):
        return
    for root, dirs, files in os.walk(kb_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(kb_dir)


def rename_kb(old: str, new: str) -> Dict[str, Any]:
    old_dir = _kb_dir(old)
    new_dir = _kb_dir(new)
    if not os.path.exists(old_dir):
        raise FileNotFoundError("Knowledge base not found")
    if os.path.exists(new_dir):
        raise FileExistsError("Target name already exists")
    os.rename(old_dir, new_dir)
    meta = get_kb(new)
    meta["name"] = new
    meta["updated_at"] = time.time()
    save_kb(meta)
    return meta


def _store_paths(kb_dir: str) -> Dict[str, str]:
    vectors_dir = os.path.join(kb_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    return {
        "index": os.path.join(vectors_dir, "index.faiss"),
        "records": os.path.join(vectors_dir, "records.jsonl"),
    }


def ingest(
    name: str,
    source_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    progress_cb: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    os.makedirs(settings.kbs_dir, exist_ok=True)
    kb_dir = _kb_dir(name)
    if os.path.exists(kb_dir):
        raise FileExistsError(f"Knowledge base already exists on disk: {kb_dir}")
    os.makedirs(kb_dir, exist_ok=True)

    meta = {
        "name": name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "created_at": time.time(),
        "updated_at": time.time(),
        "document_count": 0,
        "chunk_count": 0,
    }

    stats = ingest_into(
        name,
        source_path,
        embedding_model,
        chunk_size,
        chunk_overlap,
        top_k,
        progress_cb=progress_cb,
    )
    meta.update(stats["meta"])
    save_kb(meta)
    return {"meta": meta, **stats["stats"]}


def ingest_into(
    name: str,
    source_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    progress_cb: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    kb_dir = _kb_dir(name)
    paths = _store_paths(kb_dir)
    embedder = get_embedder(embedding_model)
    dim = embedder.get_sentence_embedding_dimension()
    store = FaissStore(dim=dim, index_path=paths["index"], records_path=paths["records"])

    processed = 0
    chunks_total = 0
    skipped = []

    all_paths = [
        file_path
        for file_path in iter_paths(source_path)
        if os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    total_files = len(all_paths)
    if progress_cb:
        progress_cb({
            "stage": "scanning",
            "total_files": total_files,
            "message": f"Found {total_files} files",
        })

    for file_path in all_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text, meta = load_document(file_path)
        except Exception as exc:
            skipped.append({"path": file_path, "reason": str(exc)})
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            continue

        vectors = embedder.encode(chunks, normalize_embeddings=False)
        vectors = np.array(vectors, dtype=np.float32)
        records = []
        for idx, chunk in enumerate(chunks):
            page = None
            if "pages" in meta:
                page = meta["pages"][0]["page"] if meta["pages"] else None
            records.append(VectorRecord(text=chunk, source=file_path, chunk_id=idx, page=page))
        store.add(vectors, records)
        processed += 1
        chunks_total += len(chunks)
        if progress_cb:
            progress_cb({
                "stage": "embedding",
                "processed_files": processed,
                "chunks": chunks_total,
                "message": f"Embedded {os.path.basename(file_path)}",
            })

    existing_docs = 0
    existing_chunks = 0
    try:
        existing = get_kb(name)
        existing_docs = existing.get("document_count", 0)
        existing_chunks = existing.get("chunk_count", 0)
    except FileNotFoundError:
        pass

    meta_update = {
        "name": name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "updated_at": time.time(),
        "document_count": existing_docs + processed,
        "chunk_count": existing_chunks + chunks_total,
    }
    return {
        "meta": meta_update,
        "stats": {
            "processed_files": processed,
            "chunks": chunks_total,
            "skipped": skipped,
        },
    }


def rebuild(name: str, source_path: str) -> Dict[str, Any]:
    meta = get_kb(name)
    delete_kb(name)
    return ingest(
        name,
        source_path,
        meta["embedding_model"],
        meta["chunk_size"],
        meta["chunk_overlap"],
        meta["top_k"],
    )
