const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function listConversations() {
  const res = await fetch(`${API_BASE}/api/conversations`);
  return res.json();
}

export async function createConversation(title, provider) {
  const res = await fetch(`${API_BASE}/api/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, provider }),
  });
  return res.json();
}

export async function deleteConversation(id) {
  await fetch(`${API_BASE}/api/conversations/${id}`, { method: "DELETE" });
}

export async function getMessages(id) {
  const res = await fetch(`${API_BASE}/api/conversations/${id}/messages`);
  return res.json();
}

export async function listKBs() {
  const res = await fetch(`${API_BASE}/api/kbs`);
  return res.json();
}

export async function createKB(payload) {
  const res = await fetch(`${API_BASE}/api/kbs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function startKBJob(payload) {
  const res = await fetch(`${API_BASE}/api/kbs/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start KB job");
  }
  return res.json();
}

export async function createKBUpload({ name, embeddingModel, chunkSize, chunkOverlap, topK, files }) {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("embedding_model", embeddingModel);
  formData.append("chunk_size", String(chunkSize));
  formData.append("chunk_overlap", String(chunkOverlap));
  formData.append("top_k", String(topK));
  files.forEach((file) => {
    formData.append("files", file, file.webkitRelativePath || file.name);
  });

  const res = await fetch(`${API_BASE}/api/kbs/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function ingestKB(name, sourcePath) {
  const res = await fetch(`${API_BASE}/api/kbs/${name}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_path: sourcePath }),
  });
  return res.json();
}

export async function rebuildKB(name, sourcePath) {
  const res = await fetch(`${API_BASE}/api/kbs/${name}/rebuild`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_path: sourcePath }),
  });
  return res.json();
}

export async function renameKB(name, newName) {
  const res = await fetch(`${API_BASE}/api/kbs/${name}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: newName }),
  });
  return res.json();
}

export async function deleteKB(name) {
  await fetch(`${API_BASE}/api/kbs/${name}`, { method: "DELETE" });
}

export async function listEmbeddingModels() {
  const res = await fetch(`${API_BASE}/api/embeddings/models`);
  return res.json();
}

export async function getSettings() {
  const res = await fetch(`${API_BASE}/api/settings`);
  return res.json();
}

export async function updateSettings(payload) {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to update settings");
  }
  return res.json();
}

export async function addEmbeddingModel(payload) {
  const res = await fetch(`${API_BASE}/api/embeddings/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function startEmbeddingJob(payload) {
  const res = await fetch(`${API_BASE}/api/embeddings/models/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start embedding job");
  }
  return res.json();
}

export async function deleteEmbeddingModel(modelId) {
  await fetch(`${API_BASE}/api/embeddings/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
  });
}

export async function pickFolder() {
  const res = await fetch(`${API_BASE}/api/pick-folder`, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Folder picker failed");
  }
  return res.json();
}

export function streamJob(jobId, onUpdate) {
  const es = new EventSource(`${API_BASE}/api/jobs/${jobId}/stream`);
  es.onmessage = (event) => {
    if (!event.data) return;
    const data = JSON.parse(event.data);
    onUpdate(data, es);
    if (data.status === "finished" || data.status === "failed") {
      es.close();
    }
  };
  es.onerror = () => {
    es.close();
  };
  return es;
}

export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to fetch job");
  }
  return res.json();
}

export async function sendChat({
  conversationId,
  message,
  kbNames,
  topK,
  ragMode,
  provider,
  onToken,
}) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message,
      kb_names: kbNames,
      top_k: topK,
      rag_mode: ragMode,
      provider,
      stream: true,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat failed");
  }
  if (!res.body) {
    throw new Error("No streaming body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.replace(/^data:\s*/, "");
      if (!payload) continue;
      const data = JSON.parse(payload);
      onToken(data);
    }
  }
}
