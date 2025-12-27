import threading
import time
import uuid
from typing import Dict, Any


class ProgressStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, kind: str) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "stage": "starting",
                "total_files": 0,
                "processed_files": 0,
                "chunks": 0,
                "message": "",
                "logs": [],
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                "error": None,
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = time.time()

    def append_log(self, job_id: str, message: str) -> None:
        if not message:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append({"ts": time.time(), "message": message})
            job["updated_at"] = time.time()

    def finish(self, job_id: str) -> None:
        self.update(job_id, status="finished", stage="done", finished_at=time.time())

    def fail(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", stage="error", error=error, finished_at=time.time())
        self.append_log(job_id, f"error: {error}")

    def get(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


progress_store = ProgressStore()
