import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from .settings import settings


@dataclass
class EmbeddingModelInfo:
    model_id: str
    path: str
    dimension: int
    size_mb: float
    tag: str | None = None


def _registry_path() -> str:
    os.makedirs(settings.embeddings_dir, exist_ok=True)
    return os.path.join(settings.embeddings_dir, "models.json")


def _load_registry() -> Dict[str, dict]:
    path = _registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(data: Dict[str, dict]) -> None:
    path = _registry_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return round(total / (1024 * 1024), 2)


def list_models() -> List[EmbeddingModelInfo]:
    registry = _load_registry()
    models = []
    for model_id, info in registry.items():
        models.append(
            EmbeddingModelInfo(
                model_id=model_id,
                path=info["path"],
                dimension=info["dimension"],
                size_mb=info["size_mb"],
                tag=info.get("tag"),
            )
        )
    return models


def register_model(model_id: str, tag: str | None = None) -> EmbeddingModelInfo:
    model_dir = os.path.join(settings.embeddings_dir, model_id.replace("/", "__"))
    os.makedirs(model_dir, exist_ok=True)
    model = SentenceTransformer(model_id, cache_folder=model_dir)
    dimension = model.get_sentence_embedding_dimension()
    size_mb = _dir_size_mb(model_dir)

    registry = _load_registry()
    registry[model_id] = {
        "path": model_dir,
        "dimension": dimension,
        "size_mb": size_mb,
        "tag": tag,
    }
    _save_registry(registry)
    return EmbeddingModelInfo(
        model_id=model_id,
        path=model_dir,
        dimension=dimension,
        size_mb=size_mb,
        tag=tag,
    )


def delete_model(model_id: str) -> None:
    registry = _load_registry()
    info = registry.get(model_id)
    if info and os.path.exists(info["path"]):
        shutil.rmtree(info["path"], ignore_errors=True)
    registry.pop(model_id, None)
    _save_registry(registry)


@lru_cache(maxsize=4)
def get_embedder(model_id: str) -> SentenceTransformer:
    registry = _load_registry()
    info = registry.get(model_id)
    if not info:
        raise ValueError("Embedding model not registered")
    return SentenceTransformer(model_id, cache_folder=info["path"])
