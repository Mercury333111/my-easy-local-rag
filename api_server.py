"""
api_server.py - FastAPI HTTP service for Easy Local RAG.

Exposes the RAG pipeline as a REST API with OpenAPI/Swagger docs.

Usage:
    python api_server.py                        # Start on default port 8000
    python api_server.py --port 8080            # Custom port
    python api_server.py --host 0.0.0.0         # Listen on all interfaces

Endpoints:
    GET  /health              Health check + system status
    POST /query               Ask a question (returns answer + sources + latency)
    POST /upload              Upload a document file
    GET  /documents           List all indexed documents
    DELETE /documents/{name}  Remove a document by name
    GET  /metrics             Return latency statistics as JSON
"""

import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

from core import (
    load_config, load_vault, load_registry, save_registry,
    load_or_generate_embeddings, BM25Index,
    hybrid_search, format_context, MetricsTracker,
)

# ===== App =====

app = FastAPI(
    title="Easy Local RAG API",
    description="Local RAG system with hybrid retrieval (BM25 + Vector) powered by Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global state (initialized on startup) =====

class AppState:
    config: dict = {}
    client: OpenAI = None
    ollama_model: str = ""
    embedding_model: str = ""
    vault_embeddings = None
    vault_content: list = []
    bm25_index: BM25Index = None
    registry: list = []
    metrics: MetricsTracker = None

state = AppState()


def reload_resources():
    """Reload vault, embeddings, BM25 index from disk."""
    state.config = load_config()
    vault_file = state.config["vault_file"]
    registry_file = state.config["registry_file"]
    cache_file = state.config["embeddings_cache_file"]

    state.registry = load_registry(registry_file)
    state.vault_embeddings, state.vault_content = load_or_generate_embeddings(
        vault_file, state.embedding_model, cache_file=cache_file
    )
    state.bm25_index = BM25Index(state.vault_content)


@app.on_event("startup")
def startup():
    state.config = load_config()
    api_cfg = state.config["ollama_api"]
    state.ollama_model = state.config["ollama_model"]
    state.embedding_model = state.config["embedding_model"]
    state.client = OpenAI(base_url=api_cfg["base_url"], api_key=api_cfg.get("api_key", ""))
    state.metrics = MetricsTracker()
    reload_resources()


# ===== Schemas =====

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    use_rewrite: bool = Field(True, description="Whether to rewrite the query for better retrieval")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override final_top_k for this query")

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: dict

class DocumentInfo(BaseModel):
    source: str
    chunk_count: int

class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    total_chunks: int
    total_documents: int
    retrieval_config: dict


# ===== Endpoints =====

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with system status."""
    from core import registry_sources
    sources = registry_sources(state.config["registry_file"])
    return HealthResponse(
        status="ok",
        model=state.ollama_model,
        embedding_model=state.embedding_model,
        total_chunks=len(state.vault_content),
        total_documents=len(sources),
        retrieval_config=state.config.get("retrieval", {}),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question and get an answer with sources and latency metrics."""
    timings = {}
    total_start = time.perf_counter()

    # Query rewrite
    if req.use_rewrite:
        t0 = time.perf_counter()
        try:
            context = ""  # no conversation history in stateless API
            rewrite_prompt = f"""Rewrite the following query to be more specific and informative for retrieval.
Return ONLY the rewritten query, nothing else.

Original query: {req.question}"""
            resp = state.client.chat.completions.create(
                model=state.ollama_model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=200,
            )
            rewritten = resp.choices[0].message.content.strip()
            query_text = rewritten if rewritten else req.question
        except Exception:
            query_text = req.question
        timings["query_rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    else:
        query_text = req.question

    # Override top_k if specified
    config_copy = dict(state.config)
    if req.top_k is not None:
        retrieval_cfg = dict(config_copy.get("retrieval", {}))
        retrieval_cfg["final_top_k"] = req.top_k
        config_copy["retrieval"] = retrieval_cfg

    # Hybrid retrieval
    t0 = time.perf_counter()
    chunks = hybrid_search(
        query_text, state.vault_embeddings, state.vault_content,
        state.bm25_index, state.embedding_model, config_copy
    )
    timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Format context
    context_str = format_context(chunks, state.registry) if chunks else ""
    max_context_chars = 4000
    if len(context_str) > max_context_chars:
        context_str = context_str[:max_context_chars] + "..."

    # Build prompt
    user_msg = req.question
    if context_str:
        user_msg = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请如实说明。

参考资料：
{context_str}

问题：{req.question}"""

    # Generate
    t0 = time.perf_counter()
    try:
        resp = state.client.chat.completions.create(
            model=state.ollama_model,
            messages=[
                {"role": "system", "content": state.config.get("system_message", "")},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2000,
        )
        answer = resp.choices[0].message.content
        if not answer or not answer.strip():
            answer = "[Model returned empty response]"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")
    timings["llm_generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 1)

    # Sources
    sources = []
    for c in chunks:
        src = {"text": c["text"][:200], "score": round(c["score"], 4), "index": c["index"]}
        if c["index"] < len(state.registry):
            src["source"] = state.registry[c["index"]].get("source", "unknown")
        sources.append(src)

    # Record metrics
    state.metrics._current = timings
    state.metrics.records.append(timings.copy())

    return QueryResponse(answer=answer, sources=sources, latency_ms=timings)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document to the RAG system."""
    # Validate format
    ext = os.path.splitext(file.filename)[1].lower()
    supported = state.config.get("supported_formats", [])
    if ext not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Supported: {supported}")

    # Save to uploads dir
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process the file using upload.py's logic
    try:
        from upload import parse_file
        text = parse_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or could not be parsed")

    # Chunk
    from core import chunk_text
    chunks = chunk_text(
        text,
        chunk_size=state.config.get("chunk_size", 300),
        overlap=state.config.get("chunk_overlap", 80),
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from file")

    # Append to vault
    vault_file = state.config["vault_file"]
    registry_file = state.config["registry_file"]

    # Remove old version if exists
    from core import remove_source_from_vault
    remove_source_from_vault(vault_file, registry_file, file.filename)

    # Append new chunks
    existing = load_vault(vault_file)
    registry = load_registry(registry_file)

    for chunk in chunks:
        existing.append(chunk)
        registry.append({"source": file.filename})

    from core import save_vault
    save_vault(vault_file, existing)
    save_registry(registry_file, registry)

    # Reload resources
    reload_resources()

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks": len(state.vault_content),
    }


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    """List all indexed documents with chunk counts."""
    from collections import Counter
    registry = load_registry(state.config["registry_file"])
    counts = Counter(e.get("source", "unknown") for e in registry)
    return [DocumentInfo(source=src, chunk_count=cnt) for src, cnt in counts.most_common()]


@app.delete("/documents/{name}")
def remove_document(name: str):
    """Remove a document by name."""
    from core import remove_source_from_vault
    vault_file = state.config["vault_file"]
    registry_file = state.config["registry_file"]

    removed, remaining = remove_source_from_vault(vault_file, registry_file, name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found")

    reload_resources()
    return {"status": "ok", "removed_chunks": removed, "remaining_chunks": remaining}


@app.get("/metrics")
def get_metrics():
    """Return latency statistics."""
    return state.metrics.get_stats()


# ===== Main =====

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Easy Local RAG API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
