import React, { useEffect, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import {
  listConversations,
  createConversation,
  deleteConversation,
  getMessages,
  listKBs,
  createKB,
  ingestKB,
  rebuildKB,
  renameKB,
  deleteKB,
  listEmbeddingModels,
  getSettings,
  updateSettings,
  deleteEmbeddingModel,
  startEmbeddingJob,
  startKBJob,
  streamJob,
  getJob,
  sendChat,
  createKBUpload,
} from "./api.js";

const PAGES = [
  { id: "chat", label: "Chat" },
  { id: "kbs", label: "Knowledge Bases" },
  { id: "create", label: "Create KB" },
  { id: "embeddings", label: "Embedding Models" },
];

export default function App() {
  const [page, setPage] = useState("chat");
  const [conversations, setConversations] = useState([]);
  const [activeConversations, setActiveConversations] = useState({
    gemini: null,
    codex: null,
  });
  const [messagesByProvider, setMessagesByProvider] = useState({
    gemini: [],
    codex: [],
  });
  const [chatInput, setChatInput] = useState("");
  const [kbs, setKbs] = useState([]);
  const [selectedKBs, setSelectedKBs] = useState([]);
  const [topK, setTopK] = useState(5);
  const [ragMode, setRagMode] = useState("hybrid");
  const [provider, setProvider] = useState("gemini");
  const [dualMode, setDualMode] = useState(false);
  const [dualComposer, setDualComposer] = useState(false);
  const [dualInputs, setDualInputs] = useState({ gemini: "", codex: "" });
  const [chatStatusByProvider, setChatStatusByProvider] = useState({
    gemini: "",
    codex: "",
  });
  const chatWindowRefs = useRef({ gemini: null, codex: null });
  const lastLoadedConvo = useRef({ gemini: null, codex: null });

  const [kbForm, setKbForm] = useState({
    name: "",
    source_path: "",
    embedding_model: "",
    chunk_size: 800,
    chunk_overlap: 100,
    top_k: 5,
  });
  const [kbStatus, setKbStatus] = useState("");
  const [kbProgress, setKbProgress] = useState(null);
  const [kbJobRunning, setKbJobRunning] = useState(false);
  const [kbJobId, setKbJobId] = useState(null);
  const [kbFiles, setKbFiles] = useState([]);

  const [embeddingModels, setEmbeddingModels] = useState([]);
  const [modelForm, setModelForm] = useState({ model_id: "", tag: "" });
  const [modelStatus, setModelStatus] = useState("");
  const [modelProgress, setModelProgress] = useState(null);
  const [embeddingDevice, setEmbeddingDevice] = useState("cpu");

  useEffect(() => {
    refreshConversations();
    refreshKBs();
    refreshEmbeddingModels();
    refreshSettings();
  }, []);

  useEffect(() => {
    if (page === "kbs") {
      refreshKBs();
    }
  }, [page]);

  useEffect(() => {
    if (!kbJobRunning || !kbJobId) return;
    const interval = setInterval(async () => {
      try {
        const snapshot = await getJob(kbJobId);
        setKbProgress(snapshot);
        if (snapshot.status === "failed") {
          setKbStatus(snapshot.error || "Ingest failed");
          setKbJobRunning(false);
        }
        if (snapshot.status === "finished") {
          setKbStatus(
            `Processed ${snapshot.processed_files || 0} files, ${snapshot.chunks || 0} chunks`
          );
          refreshKBs();
          setKbJobRunning(false);
        }
      } catch (err) {
        setKbStatus(err.message || "Failed to fetch job status");
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [kbJobRunning, kbJobId]);

  useEffect(() => {
    const convo = activeConversations.gemini;
    if (!convo) {
      setMessagesByProvider((prev) => ({ ...prev, gemini: [] }));
      return;
    }
    getMessages(convo.id).then((data) => {
      const incoming = data.messages || [];
      setMessagesByProvider((prev) => {
        if (
          lastLoadedConvo.current.gemini === convo.id &&
          prev.gemini.length > incoming.length
        ) {
          return prev;
        }
        lastLoadedConvo.current.gemini = convo.id;
        return { ...prev, gemini: incoming };
      });
    });
  }, [activeConversations.gemini]);

  useEffect(() => {
    const convo = activeConversations.codex;
    if (!convo) {
      setMessagesByProvider((prev) => ({ ...prev, codex: [] }));
      return;
    }
    getMessages(convo.id).then((data) => {
      const incoming = data.messages || [];
      setMessagesByProvider((prev) => {
        if (
          lastLoadedConvo.current.codex === convo.id &&
          prev.codex.length > incoming.length
        ) {
          return prev;
        }
        lastLoadedConvo.current.codex = convo.id;
        return { ...prev, codex: incoming };
      });
    });
  }, [activeConversations.codex]);

  useEffect(() => {
    const ref = chatWindowRefs.current.gemini;
    if (!ref) return;
    ref.scrollTop = ref.scrollHeight;
  }, [messagesByProvider.gemini]);

  useEffect(() => {
    const ref = chatWindowRefs.current.codex;
    if (!ref) return;
    ref.scrollTop = ref.scrollHeight;
  }, [messagesByProvider.codex]);

  const renderMarkdown = (text) => {
    const html = marked.parse(text || "");
    return { __html: DOMPurify.sanitize(html) };
  };

  const formatScore = (score) => {
    if (score === undefined || score === null) return "";
    return Number(score).toFixed(3);
  };

  const formatSnippet = (text) => {
    if (!text) return "";
    const cleaned = text.replace(/\s+/g, " ").trim();
    if (cleaned.length <= 280) return cleaned;
    return `${cleaned.slice(0, 280)}…`;
  };

  async function refreshConversations() {
    const data = await listConversations();
    setConversations(data.conversations || []);
  }

  async function refreshKBs() {
    const data = await listKBs();
    setKbs(data.kbs || []);
  }

  async function refreshEmbeddingModels() {
    const data = await listEmbeddingModels();
    setEmbeddingModels(data.models || []);
    if (!kbForm.embedding_model && data.models && data.models.length) {
      setKbForm((prev) => ({ ...prev, embedding_model: data.models[0].model_id }));
    }
  }

  async function refreshSettings() {
    const data = await getSettings();
    if (data.embedding_device) {
      setEmbeddingDevice(data.embedding_device);
    }
  }

  async function startNewChat() {
    if (dualMode) {
      const [geminiConvo, codexConvo] = await Promise.all([
        createConversation(null, "gemini"),
        createConversation(null, "codex"),
      ]);
      setActiveConversations({ gemini: geminiConvo, codex: codexConvo });
    } else {
      const convo = await createConversation(null, provider);
      setActiveConversations((prev) => ({ ...prev, [provider]: convo }));
    }
    setPage("chat");
    await refreshConversations();
  }

  async function removeConversation(id) {
    await deleteConversation(id);
    setActiveConversations((prev) => {
      const updated = { ...prev };
      if (updated.gemini && updated.gemini.id === id) {
        updated.gemini = null;
      }
      if (updated.codex && updated.codex.id === id) {
        updated.codex = null;
      }
      return updated;
    });
    await refreshConversations();
  }

  async function ensureConversation(providerKey) {
    const existing = activeConversations[providerKey];
    if (existing) return existing;
    const convo = await createConversation(null, providerKey);
    setActiveConversations((prev) => ({ ...prev, [providerKey]: convo }));
    return convo;
  }

  async function handleSendMessage(event, overrideProvider) {
    event.preventDefault();
    const prompt = overrideProvider
      ? dualInputs[overrideProvider].trim()
      : chatInput.trim();
    if (!prompt) return;
    if (overrideProvider) {
      setDualInputs((prev) => ({ ...prev, [overrideProvider]: "" }));
    } else {
      setChatInput("");
    }

    const providersToUse = overrideProvider
      ? [overrideProvider]
      : dualMode
        ? ["gemini", "codex"]
        : [provider];

    await Promise.all(
      providersToUse.map(async (providerKey) => {
        setChatStatusByProvider((prev) => ({ ...prev, [providerKey]: "Thinking..." }));
        const convo = await ensureConversation(providerKey);
        const newUserMessage = {
          id: `local-${Date.now()}-${providerKey}`,
          role: "user",
          content: prompt,
        };

        setMessagesByProvider((prev) => ({
          ...prev,
          [providerKey]: [
            ...prev[providerKey],
            newUserMessage,
            { role: "assistant", content: "" },
          ],
        }));

        let assistantText = "";
        let assistantSources = null;

        try {
          await sendChat({
            conversationId: convo.id,
            message: newUserMessage.content,
            kbNames: selectedKBs,
            topK,
            ragMode,
            provider: providerKey,
            onToken: (data) => {
              if (data.type === "delta") {
                assistantText += data.content || "";
                setMessagesByProvider((prev) => {
                  const updated = [...prev[providerKey]];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: assistantText,
                    sources: assistantSources,
                  };
                  return { ...prev, [providerKey]: updated };
                });
              }
              if (data.type === "done") {
                assistantSources = data.sources || null;
                setChatStatusByProvider((prev) => ({ ...prev, [providerKey]: "" }));
                setMessagesByProvider((prev) => {
                  const updated = [...prev[providerKey]];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: assistantText,
                    sources: assistantSources,
                  };
                  return { ...prev, [providerKey]: updated };
                });
                refreshConversations();
              }
              if (data.type === "error") {
                setChatStatusByProvider((prev) => ({
                  ...prev,
                  [providerKey]: data.message || "Streaming error",
                }));
              }
            },
          });
        } catch (err) {
          setChatStatusByProvider((prev) => ({
            ...prev,
            [providerKey]: err.message || "Chat failed",
          }));
          setMessagesByProvider((prev) => {
            const updated = [...prev[providerKey]];
            updated[updated.length - 1] = {
              role: "assistant",
              content: `Error: ${err.message || "Chat failed"}`,
            };
            return { ...prev, [providerKey]: updated };
          });
        }
      })
    );
  }

  async function handleCreateKB(event) {
    event.preventDefault();
    setKbStatus("Starting...");
    setKbProgress(null);
    setKbJobRunning(true);
    try {
      if (!kbFiles.length && !kbForm.source_path.trim()) {
        setKbStatus("Provide a source path or select a folder.");
        setKbJobRunning(false);
        return;
      }
      if (kbFiles.length) {
        const result = await createKBUpload({
          name: kbForm.name,
          embeddingModel: kbForm.embedding_model,
          chunkSize: kbForm.chunk_size,
          chunkOverlap: kbForm.chunk_overlap,
          topK: kbForm.top_k,
          files: kbFiles,
        });
        setKbStatus(`Processed ${result.processed_files} files, ${result.chunks} chunks`);
        refreshKBs();
        setKbJobRunning(false);
      } else {
        const job = await startKBJob(kbForm);
        setKbJobId(job.job_id);
        try {
          const snapshot = await getJob(job.job_id);
          setKbProgress(snapshot);
          if (snapshot.status === "failed") {
            setKbStatus(snapshot.error || "Ingest failed");
            setKbJobRunning(false);
            return;
          }
          if (snapshot.status === "finished") {
            setKbStatus(
              `Processed ${snapshot.processed_files || 0} files, ${snapshot.chunks || 0} chunks`
            );
            refreshKBs();
            setKbJobRunning(false);
            return;
          }
        } catch (err) {
          setKbStatus(err.message || "Failed to fetch job status");
        }
        streamJob(job.job_id, (data) => {
          setKbProgress(data);
          if (data.status === "failed") {
            setKbStatus(data.error || "Ingest failed");
            setKbJobRunning(false);
          }
          if (data.status === "finished") {
            setKbStatus(
              `Processed ${data.processed_files || 0} files, ${data.chunks || 0} chunks`
            );
            refreshKBs();
            setKbJobRunning(false);
          }
        });
        setKbFiles([]);
      }
      setKbForm({
        name: "",
        source_path: "",
        embedding_model: kbForm.embedding_model,
        chunk_size: 800,
        chunk_overlap: 100,
        top_k: 5,
      });
      setKbFiles([]);
    } catch (err) {
      setKbStatus(err.message || "Failed to ingest");
    }
  }

  async function handleAddModel(event) {
    event.preventDefault();
    setModelStatus("Starting...");
    setModelProgress(null);
    try {
      const job = await startEmbeddingJob(modelForm);
      streamJob(job.job_id, (data) => {
        setModelProgress(data);
        if (data.status === "failed") {
          setModelStatus(data.error || "Model download failed");
        }
        if (data.status === "finished") {
          setModelStatus("Model added");
          refreshEmbeddingModels();
        }
      });
      setModelForm({ model_id: "", tag: "" });
    } catch (err) {
      setModelStatus(err.message || "Failed to add model");
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <h1>Local RAG Studio</h1>
            <p>Open WebUI-style chat with local knowledge bases</p>
          </div>
        </div>
        <nav className="nav">
          {PAGES.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "nav-item active" : "nav-item"}
              onClick={() => setPage(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-section">
          <div className="section-header">
            <h2>Conversations</h2>
            <button className="ghost" onClick={startNewChat}>New Chat</button>
          </div>
          <div className="conversation-list">
            <div className="conversation-group">
              <div className="group-title">Gemini</div>
              {conversations
                .filter((convo) => (convo.provider || "gemini") === "gemini")
                .map((convo) => (
                  <div
                    key={convo.id}
                    className={
                      activeConversations.gemini && activeConversations.gemini.id === convo.id
                        ? "conversation-item active"
                        : "conversation-item"
                    }
                  >
                    <button
                      className="conversation-link"
                      onClick={() => {
                        setActiveConversations((prev) => ({ ...prev, gemini: convo }));
                        setProvider("gemini");
                        setPage("chat");
                      }}
                    >
                      {convo.title || "New chat"}
                    </button>
                    <button className="ghost" onClick={() => removeConversation(convo.id)}>
                      ✕
                    </button>
                  </div>
                ))}
            </div>
            <div className="conversation-group">
              <div className="group-title">Codex</div>
              {conversations
                .filter((convo) => (convo.provider || "gemini") === "codex")
                .map((convo) => (
                  <div
                    key={convo.id}
                    className={
                      activeConversations.codex && activeConversations.codex.id === convo.id
                        ? "conversation-item active"
                        : "conversation-item"
                    }
                  >
                    <button
                      className="conversation-link"
                      onClick={() => {
                        setActiveConversations((prev) => ({ ...prev, codex: convo }));
                        setProvider("codex");
                        setPage("chat");
                      }}
                    >
                      {convo.title || "New chat"}
                    </button>
                    <button className="ghost" onClick={() => removeConversation(convo.id)}>
                      ✕
                    </button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </aside>

      <main className="main">
        {page === "chat" && (
          <section className="panel">
            <header className="panel-header">
              <div>
                <h2>Chat</h2>
                <p>Pick knowledge bases to ground responses.</p>
              </div>
              <div className="kb-select">
                <label>
                  Top K
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                  />
                </label>
                <label>
                  Mode
                  <select value={ragMode} onChange={(event) => setRagMode(event.target.value)}>
                    <option value="hybrid">Hybrid chat</option>
                    <option value="rag_only">RAG only</option>
                  </select>
                </label>
                <label>
                  Provider
                  <select
                    value={provider}
                    onChange={(event) => setProvider(event.target.value)}
                    disabled={dualMode}
                  >
                    <option value="codex">Codex</option>
                    <option value="gemini">Gemini</option>
                  </select>
                </label>
                <label>
                  Dual
                  <input
                    type="checkbox"
                    checked={dualMode}
                    onChange={(event) => setDualMode(event.target.checked)}
                  />
                </label>
                <label>
                  Dual input
                  <input
                    type="checkbox"
                    checked={dualComposer}
                    onChange={(event) => setDualComposer(event.target.checked)}
                    disabled={!dualMode}
                  />
                </label>
              </div>
            </header>

            <div className="kb-chip-row">
              {kbs.length === 0 && <span className="muted">No KBs yet</span>}
              {kbs.map((kb) => (
                <label key={kb.name} className="kb-chip">
                  <input
                    type="checkbox"
                    checked={selectedKBs.includes(kb.name)}
                    onChange={(event) => {
                      if (event.target.checked) {
                        setSelectedKBs((prev) => [...prev, kb.name]);
                      } else {
                        setSelectedKBs((prev) => prev.filter((name) => name !== kb.name));
                      }
                    }}
                  />
                  {kb.name}
                </label>
              ))}
            </div>

            <div className={dualMode ? "chat-split" : ""}>
              {(dualMode ? ["gemini", "codex"] : [provider]).map((providerKey) => (
                <div key={providerKey} className="chat-pane">
                  <div className="pane-header">
                    <span>{providerKey === "gemini" ? "Gemini" : "Codex"}</span>
                    {chatStatusByProvider[providerKey] && (
                      <span className="status live">
                        <span className="pulse-dot" />
                        <span className="pulse-text">{chatStatusByProvider[providerKey]}</span>
                      </span>
                    )}
                  </div>
                  <div
                    className="chat-window"
                    ref={(node) => {
                      chatWindowRefs.current[providerKey] = node;
                    }}
                  >
                    {messagesByProvider[providerKey].length === 0 && (
                      <div className="empty">Start a conversation to see messages here.</div>
                    )}
                    {messagesByProvider[providerKey].map((msg, index) => (
                      <div key={index} className={`message ${msg.role}`}>
                        <div className="message-meta">
                          {msg.role === "user"
                            ? "You"
                            : providerKey === "gemini"
                              ? "Gemini"
                              : "Codex"}
                        </div>
                        {msg.role === "assistant" ? (
                          <div
                            className="message-text markdown"
                            dangerouslySetInnerHTML={renderMarkdown(msg.content)}
                          />
                        ) : (
                          <div className="message-text">{msg.content}</div>
                        )}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="citations">
                            {msg.sources.slice(0, 5).map((source, idx) => (
                              <div key={idx} className="citation">
                                <span className="citation-label">
                                  {source.source}
                                  {source.page ? ` · page ${source.page}` : ` · chunk ${source.chunk_id}`}
                                  {source.score !== undefined && source.score !== null
                                    ? ` · score ${formatScore(source.score)}`
                                    : ""}
                                </span>
                                <div className="citation-tooltip">
                                  <div className="tooltip-title">Excerpt</div>
                                  <div className="tooltip-text">{formatSnippet(source.text)}</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {dualMode && dualComposer ? (
              <div className="dual-composer">
                {["gemini", "codex"].map((providerKey) => (
                  <form
                    key={providerKey}
                    className="composer"
                    onSubmit={(event) => handleSendMessage(event, providerKey)}
                  >
                    <textarea
                      value={dualInputs[providerKey]}
                      onChange={(event) =>
                        setDualInputs((prev) => ({
                          ...prev,
                          [providerKey]: event.target.value,
                        }))
                      }
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          handleSendMessage(event, providerKey);
                        }
                      }}
                      placeholder={`Ask ${providerKey}...`}
                    />
                    <button type="submit">Send</button>
                  </form>
                ))}
              </div>
            ) : (
              <form className="composer" onSubmit={handleSendMessage}>
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleSendMessage(event);
                    }
                  }}
                  placeholder="Ask about your documents or chat freely..."
                />
                <button type="submit">Send</button>
              </form>
            )}
          </section>
        )}

        {page === "kbs" && (
          <section className="panel">
            <header className="panel-header">
              <div>
                <h2>Knowledge Bases</h2>
                <p>Manage existing knowledge bases.</p>
              </div>
              <button
                className="ghost"
                onClick={() => {
                  refreshKBs();
                }}
              >
                Refresh
              </button>
            </header>
            <div className="card-grid">
              {kbs.map((kb) => (
                <div key={kb.name} className="card">
                  <h3>{kb.name}</h3>
                  <p>Embedding: {kb.embedding_model}</p>
                  <p>Chunks: {kb.chunk_count} · Docs: {kb.document_count}</p>
                  <p>Chunk size: {kb.chunk_size} / overlap {kb.chunk_overlap}</p>
                  <div className="card-actions">
                    <button
                      className="ghost"
                      onClick={async () => {
                        const sourcePath = window.prompt("Path to ingest additional docs");
                        if (sourcePath) {
                          await ingestKB(kb.name, sourcePath);
                          refreshKBs();
                        }
                      }}
                    >
                      Add docs
                    </button>
                    <button
                      className="ghost"
                      onClick={async () => {
                        const newName = window.prompt("New name", kb.name);
                        if (newName && newName !== kb.name) {
                          await renameKB(kb.name, newName);
                          refreshKBs();
                        }
                      }}
                    >
                      Rename
                    </button>
                    <button
                      className="ghost"
                      onClick={async () => {
                        const sourcePath = window.prompt("Path to rebuild KB");
                        if (sourcePath) {
                          await rebuildKB(kb.name, sourcePath);
                          refreshKBs();
                        }
                      }}
                    >
                      Rebuild
                    </button>
                    <button
                      className="danger"
                      onClick={async () => {
                        if (window.confirm("Delete this KB?")) {
                          await deleteKB(kb.name);
                          refreshKBs();
                        }
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {page === "create" && (
          <section className="panel">
            <header className="panel-header">
              <div>
                <h2>Create Knowledge Base</h2>
                <p>Configure chunking, retrieval, and embedding model.</p>
              </div>
            </header>
            <form className="form" onSubmit={handleCreateKB}>
              <label>
                KB name
                <input
                  value={kbForm.name}
                  onChange={(event) => setKbForm({ ...kbForm, name: event.target.value })}
                  required
                />
              </label>
              <label>
                Source path
                <input
                  value={kbForm.source_path}
                  onChange={(event) => setKbForm({ ...kbForm, source_path: event.target.value })}
                  placeholder="/home/you/docs"
                />
              </label>
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  const input = window.prompt(
                    "Paste a Windows path to convert (e.g. C:\\\\Users\\\\you\\\\Docs)"
                  );
                  if (!input) return;
                  const trimmed = input.trim().replace(/^["']|["']$/g, "");
                  const match = trimmed.match(/^([A-Za-z]):[\\/](.*)$/);
                  if (!match) {
                    setKbStatus("Invalid Windows path.");
                    return;
                  }
                  const drive = match[1].toLowerCase();
                  const rest = match[2].replace(/[\\]+/g, "/");
                  const wslPath = `/mnt/${drive}/${rest}`;
                  setKbForm((prev) => ({ ...prev, source_path: wslPath }));
                  setKbStatus(`WSL path: ${wslPath}`);
                }}
              >
                WSL path helper
              </button>
              <label>
                Upload folder
                <input
                  type="file"
                  webkitdirectory="true"
                  onChange={(event) => setKbFiles(Array.from(event.target.files || []))}
                />
              </label>
              <label>
                Embedding model
                <select
                  value={kbForm.embedding_model}
                  onChange={(event) => setKbForm({ ...kbForm, embedding_model: event.target.value })}
                  required
                >
                  <option value="" disabled>Select model</option>
                  {embeddingModels.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_id}
                    </option>
                  ))}
                </select>
              </label>
              <div className="form-row">
                <label>
                  Chunk size
                  <input
                    type="number"
                    value={kbForm.chunk_size}
                    onChange={(event) => setKbForm({ ...kbForm, chunk_size: Number(event.target.value) })}
                  />
                </label>
                <label>
                  Chunk overlap
                  <input
                    type="number"
                    value={kbForm.chunk_overlap}
                    onChange={(event) => setKbForm({ ...kbForm, chunk_overlap: Number(event.target.value) })}
                  />
                </label>
                <label>
                  Top K
                  <input
                    type="number"
                    value={kbForm.top_k}
                    onChange={(event) => setKbForm({ ...kbForm, top_k: Number(event.target.value) })}
                  />
                </label>
              </div>
              <button type="submit" disabled={kbJobRunning}>
                {kbJobRunning ? "Building..." : "Build Knowledge Base"}
              </button>
            </form>
            {kbStatus && <div className="status">{kbStatus}</div>}
            {kbProgress && (
              <div className="status">
                Stage: {kbProgress.stage} · Files: {kbProgress.processed_files || 0}/
                {kbProgress.total_files || 0} · Chunks: {kbProgress.chunks || 0}
              </div>
            )}
            {kbProgress && kbProgress.logs && kbProgress.logs.length ? (
              <div className="log-panel">
                {kbProgress.logs.slice(-10).map((entry) => (
                  <div key={entry.ts} className="log-line">
                    {entry.message}
                  </div>
                ))}
              </div>
            ) : null}
            {kbProgress && kbProgress.status === "failed" && kbProgress.error ? (
              <div className="status">Error: {kbProgress.error}</div>
            ) : null}
          </section>
        )}

        {page === "embeddings" && (
          <section className="panel">
            <header className="panel-header">
              <div>
                <h2>Embedding Models</h2>
                <p>Download and manage local sentence-transformers.</p>
              </div>
              <button className="ghost" onClick={refreshEmbeddingModels}>Refresh</button>
            </header>
            <div className="settings-row">
              <label>
                Embedding device
                <select
                  value={embeddingDevice}
                  onChange={async (event) => {
                    const value = event.target.value;
                    setEmbeddingDevice(value);
                    try {
                      await updateSettings({ embedding_device: value });
                    } catch (err) {
                      setModelStatus(err.message || "Failed to update device");
                    }
                  }}
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                </select>
              </label>
            </div>
            <div className="card-grid">
              {embeddingModels.map((model) => (
                <div key={model.model_id} className="card">
                  <h3>{model.model_id}</h3>
                  <p>Dimension: {model.dimension}</p>
                  <p>Size: {model.size_mb} MB</p>
                  {model.tag && <p>Tag: {model.tag}</p>}
                  <button
                    className="danger"
                    onClick={async () => {
                      if (window.confirm("Delete this model?")) {
                        await deleteEmbeddingModel(model.model_id);
                        refreshEmbeddingModels();
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
            <form className="form" onSubmit={handleAddModel}>
              <label>
                Hugging Face repo ID
                <input
                  value={modelForm.model_id}
                  onChange={(event) => setModelForm({ ...modelForm, model_id: event.target.value })}
                  placeholder="sentence-transformers/all-MiniLM-L6-v2"
                  required
                />
              </label>
              <label>
                Tag (optional)
                <input
                  value={modelForm.tag}
                  onChange={(event) => setModelForm({ ...modelForm, tag: event.target.value })}
                  placeholder="fast / multilingual / quality"
                />
              </label>
              <button type="submit">Download Model</button>
              {modelStatus && <div className="status">{modelStatus}</div>}
              {modelProgress && (
                <div className="status">
                  Stage: {modelProgress.stage} {modelProgress.message || ""}
                </div>
              )}
              {modelProgress && modelProgress.logs && modelProgress.logs.length ? (
                <div className="log-panel">
                  {modelProgress.logs.slice(-8).map((entry) => (
                    <div key={entry.ts} className="log-line">
                      {entry.message}
                    </div>
                  ))}
                </div>
              ) : null}
            </form>
          </section>
        )}
      </main>
    </div>
  );
}
