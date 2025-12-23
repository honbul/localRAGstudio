import React, { useEffect, useMemo, useRef, useState } from "react";
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
  addEmbeddingModel,
  deleteEmbeddingModel,
  sendChat,
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
  const [activeConversation, setActiveConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [kbs, setKbs] = useState([]);
  const [selectedKBs, setSelectedKBs] = useState([]);
  const [topK, setTopK] = useState(5);
  const [ragMode, setRagMode] = useState("hybrid");
  const [provider, setProvider] = useState("gemini");
  const [chatStatus, setChatStatus] = useState("");
  const chatWindowRef = useRef(null);

  const [kbForm, setKbForm] = useState({
    name: "",
    source_path: "",
    embedding_model: "",
    chunk_size: 800,
    chunk_overlap: 100,
    top_k: 5,
  });
  const [kbStatus, setKbStatus] = useState("");

  const [embeddingModels, setEmbeddingModels] = useState([]);
  const [modelForm, setModelForm] = useState({ model_id: "", tag: "" });
  const [modelStatus, setModelStatus] = useState("");

  useEffect(() => {
    refreshConversations();
    refreshKBs();
    refreshEmbeddingModels();
  }, []);

  useEffect(() => {
    if (!activeConversation) {
      setMessages([]);
      return;
    }
    getMessages(activeConversation.id).then((data) => {
      setMessages(data.messages || []);
    });
  }, [activeConversation]);

  useEffect(() => {
    if (!chatWindowRef.current) return;
    chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
  }, [messages]);

  const kbMap = useMemo(() => {
    const map = new Map();
    kbs.forEach((kb) => map.set(kb.name, kb));
    return map;
  }, [kbs]);

  const renderMarkdown = (text) => {
    const html = marked.parse(text || "");
    return { __html: DOMPurify.sanitize(html) };
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

  async function startNewChat() {
    const convo = await createConversation(null, provider);
    setActiveConversation(convo);
    setPage("chat");
    await refreshConversations();
  }

  async function removeConversation(id) {
    await deleteConversation(id);
    if (activeConversation && activeConversation.id === id) {
      setActiveConversation(null);
      setMessages([]);
    }
    await refreshConversations();
  }

  async function handleSendMessage(event) {
    event.preventDefault();
    if (!chatInput.trim()) return;
    setChatStatus("Thinking...");

    let convo = activeConversation;
    if (!convo) {
      convo = await createConversation(null, provider);
      setActiveConversation(convo);
    }

    const newUserMessage = {
      id: `local-${Date.now()}`,
      role: "user",
      content: chatInput.trim(),
    };

    setMessages((prev) => [...prev, newUserMessage, { role: "assistant", content: "" }]);
    setChatInput("");

    let assistantText = "";
    let assistantSources = null;

    try {
      await sendChat({
        conversationId: convo.id,
        message: newUserMessage.content,
        kbNames: selectedKBs,
        topK,
        ragMode,
        provider,
        onToken: (data) => {
          if (data.type === "delta") {
            assistantText += data.content || "";
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: "assistant",
                content: assistantText,
                sources: assistantSources,
              };
              return updated;
            });
          }
          if (data.type === "done") {
            assistantSources = data.sources || null;
            setChatStatus("");
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: "assistant",
                content: assistantText,
                sources: assistantSources,
              };
              return updated;
            });
            refreshConversations();
          }
          if (data.type === "error") {
            setChatStatus(data.message || "Streaming error");
          }
        },
      });
    } catch (err) {
      setChatStatus(err.message || "Chat failed");
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message || "Chat failed"}`,
        };
        return updated;
      });
    }
  }

  async function handleCreateKB(event) {
    event.preventDefault();
    setKbStatus("Ingesting...");
    try {
      const result = await createKB(kbForm);
      setKbStatus(`Processed ${result.processed_files} files, ${result.chunks} chunks`);
      setKbForm({
        name: "",
        source_path: "",
        embedding_model: kbForm.embedding_model,
        chunk_size: 800,
        chunk_overlap: 100,
        top_k: 5,
      });
      refreshKBs();
    } catch (err) {
      setKbStatus(err.message || "Failed to ingest");
    }
  }

  async function handleAddModel(event) {
    event.preventDefault();
    setModelStatus("Downloading...");
    try {
      await addEmbeddingModel(modelForm);
      setModelStatus("Model added");
      setModelForm({ model_id: "", tag: "" });
      refreshEmbeddingModels();
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
            {conversations.map((convo) => (
              <div
                key={convo.id}
                className={
                  activeConversation && activeConversation.id === convo.id
                    ? "conversation-item active"
                    : "conversation-item"
                }
              >
                <button
                  className="conversation-link"
                  onClick={() => {
                    setActiveConversation(convo);
                    setProvider(convo.provider || "gemini");
                    setPage("chat");
                  }}
                >
                  {convo.title || "New chat"}
                </button>
                <span className="provider-tag">{(convo.provider || "gemini").toUpperCase()}</span>
                <button
                  className="ghost"
                  onClick={() => removeConversation(convo.id)}
                >
                  ✕
                </button>
              </div>
            ))}
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
                  <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                    <option value="codex">Codex</option>
                    <option value="gemini">Gemini</option>
                  </select>
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

            <div className="chat-window" ref={chatWindowRef}>
              {messages.length === 0 && (
                <div className="empty">Start a conversation to see messages here.</div>
              )}
              {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.role}`}>
                  <div className="message-meta">{msg.role === "user" ? "You" : "Codex"}</div>
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
                          {source.source} · chunk {source.chunk_id}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

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
            {chatStatus && <div className="status">{chatStatus}</div>}
          </section>
        )}

        {page === "kbs" && (
          <section className="panel">
            <header className="panel-header">
              <div>
                <h2>Knowledge Bases</h2>
                <p>Manage existing knowledge bases.</p>
              </div>
              <button className="ghost" onClick={refreshKBs}>Refresh</button>
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
                  required
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
              <button type="submit">Build Knowledge Base</button>
            </form>
            {kbStatus && <div className="status">{kbStatus}</div>}
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
            </form>
          </section>
        )}
      </main>
    </div>
  );
}
