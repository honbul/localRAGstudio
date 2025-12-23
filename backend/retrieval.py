import os
from typing import List, Dict, Any

import numpy as np

from .embeddings import get_embedder
from .kb_manager import get_kb
from .vector_store import FaissStore
from .settings import settings


def _store_paths(kb_name: str) -> Dict[str, str]:
    kb_dir = os.path.join(settings.kbs_dir, kb_name.replace(" ", "_"))
    vectors_dir = os.path.join(kb_dir, "vectors")
    return {
        "index": os.path.join(vectors_dir, "index.faiss"),
        "records": os.path.join(vectors_dir, "records.jsonl"),
    }


def retrieve(kb_names: List[str], query: str, top_k: int) -> List[Dict[str, Any]]:
    results = []
    for name in kb_names:
        meta = get_kb(name)
        embedder = get_embedder(meta["embedding_model"])
        dim = embedder.get_sentence_embedding_dimension()
        query_vec = np.array(embedder.encode([query])[0], dtype=np.float32)
        paths = _store_paths(name)
        store = FaissStore(dim=dim, index_path=paths["index"], records_path=paths["records"])
        hits = store.search(query_vec, top_k)
        for hit in hits:
            hit["kb"] = name
        results.extend(hits)
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:top_k]


def build_context(results: List[Dict[str, Any]]) -> str:
    parts = []
    for item in results:
        source = item.get("source", "")
        page = item.get("page")
        kb_name = item.get("kb")
        label = f"{source}"
        if page:
            label += f" (page {page})"
        if kb_name:
            label = f"{kb_name} | {label}"
        parts.append(f"Source: {label}\n{item.get('text', '')}")
    return "\n\n".join(parts)
