import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import faiss
import numpy as np


@dataclass
class VectorRecord:
    text: str
    source: str
    chunk_id: int
    page: int | None = None


class FaissStore:
    def __init__(self, dim: int, index_path: str, records_path: str) -> None:
        self.dim = dim
        self.index_path = index_path
        self.records_path = records_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    def _load_index(self) -> faiss.Index:
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        return faiss.IndexFlatIP(self.dim)

    def _save_index(self, index: faiss.Index) -> None:
        faiss.write_index(index, self.index_path)

    def add(self, vectors: np.ndarray, records: List[VectorRecord]) -> None:
        index = self._load_index()
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self._save_index(index)

        with open(self.records_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")

    def search(self, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        index = self._load_index()
        if index.ntotal == 0:
            return []
        query_vec = query.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        records = []
        if os.path.exists(self.records_path):
            with open(self.records_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(records):
                continue
            record = records[idx]
            results.append({
                "score": float(score),
                "text": record["text"],
                "source": record["source"],
                "chunk_id": record["chunk_id"],
                "page": record.get("page"),
            })
        return results
