# RAG in Local RAG Studio

This document explains how retrieval‑augmented generation (RAG) is created and used in the app.

## Overview

The RAG flow has four stages:

1) Ingest documents into a knowledge base (KB)
2) Chunk documents into passages
3) Embed passages and store them in a local vector index
4) Retrieve top‑k chunks at chat time and inject them into the prompt

Everything runs locally. The LLM (Codex or Gemini CLI) is stateless; the app sends full chat history plus retrieved context each request.

## Knowledge base creation

When you create a KB, the backend:

- Walks the provided path or uploaded folder
- Reads supported file types: PDF, DOCX, TXT, MD
- Splits text into overlapping chunks
- Computes embeddings via sentence‑transformers
- Stores vectors in FAISS and writes metadata to disk

Each KB is stored under:

```
data/kbs/<kb_name>/
  ├── metadata.json
  └── vectors/
      ├── index.faiss
      └── records.jsonl
```

`metadata.json` records the embedding model, chunk size/overlap, document count, and last updated time.

## Embeddings

Embedding models are managed independently from KBs:

- Models are downloaded from Hugging Face by repo ID
- Cached under `data/embeddings/`
- Reused across KBs

When a KB is created, you choose which embedding model to use. That model is used for all future ingests into the same KB unless you rebuild.

## Retrieval

At chat time:

1) The user selects one or more KBs
2) The query is embedded using each KB’s embedding model
3) FAISS retrieves top‑k similar chunks per KB
4) Results are merged and re‑ranked by similarity
5) The top‑k chunks are formatted into a context block

Example context format:

```
Source: <kb_name> | /path/to/file.pdf (page 2)
<chunk text>
```

## Prompt injection

The context is prepended as a system message. The rest of the conversation history is appended in order. Two modes are supported:

- **Hybrid**: use context if relevant; answer normally for general questions
- **RAG only**: answer strictly from context

## Incremental ingest and rebuild

- **Add docs**: appends new vectors and records to the existing FAISS index
- **Rebuild**: deletes the KB and rebuilds from the specified source path

## Citations

Retrieved chunks are returned with `source`, `chunk_id`, and optional `page` and `kb` labels. The UI shows a short list of citations under assistant messages.
