from typing import List, Dict

ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def build_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        label = ROLE_LABELS.get(role, role.title())
        parts.append(f"{label}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
