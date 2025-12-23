import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

from .db import db_conn


def _now() -> str:
    return datetime.utcnow().isoformat()


def list_conversations() -> List[Dict[str, Any]]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, provider, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def create_conversation(title: str | None = None, provider: str | None = None) -> Dict[str, Any]:
    convo_id = str(uuid.uuid4())
    now = _now()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, provider, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (convo_id, title, provider, now, now),
        )
    return {"id": convo_id, "title": title, "created_at": now, "updated_at": now}


def delete_conversation(convo_id: str) -> None:
    with db_conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (convo_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (convo_id,))


def get_messages(convo_id: str) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (convo_id,),
        ).fetchall()
    messages = []
    for row in rows:
        msg = dict(row)
        if msg.get("sources"):
            msg["sources"] = json.loads(msg["sources"])
        messages.append(msg)
    return messages


def add_message(convo_id: str, role: str, content: str, sources: list | None = None) -> Dict[str, Any]:
    msg_id = str(uuid.uuid4())
    now = _now()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, convo_id, role, content, json.dumps(sources) if sources else None, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, convo_id),
        )
    return {
        "id": msg_id,
        "conversation_id": convo_id,
        "role": role,
        "content": content,
        "sources": sources,
        "created_at": now,
    }


def ensure_title(convo_id: str, content: str) -> None:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (convo_id,),
        ).fetchone()
        if row and row["title"]:
            return
        title = content.strip().split("\n")[0][:60]
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title or "New chat", convo_id),
        )


def set_provider(convo_id: str, provider: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE conversations SET provider = ? WHERE id = ?",
            (provider, convo_id),
        )
