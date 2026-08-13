"""
core.py - Shared utilities for the RAG system
Provides: config loading, vault operations, embedding cache, text chunking, hybrid retrieval, metrics
"""

import os
import json
import hashlib
import re
import time
from contextlib import contextmanager
from collections import defaultdict

import yaml
import torch
import ollama
import jieba
from rank_bm25 import BM25Okapi

# ===== File paths =====
CONFIG_FILE = "config.yaml"


def load_config():
    """Load configuration from config.yaml with defaults."""
    defaults = {
        "vault_file": "vault.txt",
        "registry_file": "vault_registry.json",
        "embeddings_cache_file": "vault_embeddings_cache.json",
        "ollama_model": "qwen3.6",
        "embedding_model": "nomic-embed-text",
        "top_k": 3,
        "similarity_threshold": 0.3,
        "chunk_size": 300,
        "chunk_overlap": 80,
        "system_message": "You are a helpful assistant.",
        "ollama_api": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "qwen3.6",
        },
        "supported_formats": [".pdf", ".txt", ".json", ".md", ".docx", ".html", ".htm", ".csv", ".xlsx"],
        "retrieval": {
            "bm25_weight": 0.4,
            "vector_weight": 0.6,
            "bm25_top_k": 10,
            "vector_top_k": 10,
            "final_top_k": 3,
        },
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg)
    return defaults


# ===== Vault operations =====

def load_vault(vault_file):
    """Load vault content lines."""
    if not os.path.exists(vault_file):
        return []
    with open(vault_file, "r", encoding="utf-8") as f:
        return f.readlines()


def save_vault(vault_file, lines):
    """Write all lines to vault file."""
    with open(vault_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def vault_hash(vault_file):
    """Compute SHA256 hash of vault content for cache invalidation."""
    if not os.path.exists(vault_file):
        return ""
    with open(vault_file, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ===== Document registry =====

def load_registry(registry_file):
    """Load the document registry."""
    if not os.path.exists(registry_file):
        return []
    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry_file, registry):
    """Save the document registry."""
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def registry_sources(registry_file):
    """Return a set of unique source file names in the registry."""
    registry = load_registry(registry_file)
    return set(entry.get("source", "unknown") for entry in registry)


def remove_source_from_vault(vault_file, registry_file, source_name):
    """Remove all chunks from a specific source file.
    Returns (removed_count, remaining_count)."""
    registry = load_registry(registry_file)
    vault_lines = load_vault(vault_file)

    indices_to_remove = set()
    new_registry = []
    for i, entry in enumerate(registry):
        if entry.get("source") == source_name:
            indices_to_remove.add(i)
        else:
            new_registry.append(entry)

    new_vault = [line for i, line in enumerate(vault_lines) if i not in indices_to_remove]

    save_vault(vault_file, new_vault)
    save_registry(registry_file, new_registry)

    removed = len(indices_to_remove)
    remaining = len(new_vault)
    return removed, remaining


def clear_vault(vault_file, registry_file):
    """Clear all vault content and registry."""
    save_vault(vault_file, [])
    save_registry(registry_file, [])


# ===== Chinese Tokenization =====

def tokenize_chinese(text):
    """Tokenize Chinese text using jieba. Returns list of tokens."""
    # Use jieba to segment Chinese, keep English words as-is
    tokens = jieba.lcut(text)
    # Filter out single-char punctuation and whitespace
    tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
    return tokens


# ===== BM25 Index =====

class BM25Index:
    """BM25 index for keyword-based retrieval over vault chunks."""

    def __init__(self, vault_content):
        self.vault_content = vault_content
        if vault_content:
            tokenized_corpus = [tokenize_chinese(line) for line in vault_content]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query, top_k=10):
        """Search for relevant chunks using BM25.
        Returns list of (index, score) tuples sorted by score descending."""
        if not self.bm25 or not self.vault_content:
            return []
        query_tokens = tokenize_chinese(query)
        scores = self.bm25.get_scores(query_tokens)
        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]


# ===== Embedding cache =====

def load_or_generate_embeddings(vault_file, embedding_model, cache_file="vault_embeddings_cache.json", force=False):
    """Load embeddings from cache, or regenerate if vault changed.
    Returns (tensor, vault_content_lines)."""

    vault_content = load_vault(vault_file)
    if not vault_content:
        return torch.tensor([]), []

    current_hash = vault_hash(vault_file)

    # Try loading cache
    if not force and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("hash") == current_hash and cache.get("model") == embedding_model:
            print(f"[Cache] Loaded {len(cache['embeddings'])} embeddings from cache")
            return torch.tensor(cache["embeddings"]), vault_content

    # Generate embeddings
    print(f"[Embedding] Generating embeddings for {len(vault_content)} chunks...")
    embeddings = []
    for i, content in enumerate(vault_content):
        response = ollama.embeddings(model=embedding_model, prompt=content)
        embeddings.append(response["embedding"])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(vault_content)}]")

    # Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"hash": current_hash, "model": embedding_model, "embeddings": embeddings}, f)
    print(f"[Cache] Saved embeddings to {cache_file}")

    return torch.tensor(embeddings), vault_content


# ===== Hybrid Retrieval =====

def hybrid_search(query, vault_embeddings, vault_content, bm25_index,
                  embedding_model, config):
    """Hybrid retrieval combining BM25 keyword search and vector similarity.
    Uses Reciprocal Rank Fusion (RRF) to merge results.

    Returns list of (chunk_text, score, source) tuples.
    """
    retrieval_cfg = config.get("retrieval", {})
    bm25_weight = retrieval_cfg.get("bm25_weight", 0.4)
    vector_weight = retrieval_cfg.get("vector_weight", 0.6)
    bm25_top_k = retrieval_cfg.get("bm25_top_k", 10)
    vector_top_k = retrieval_cfg.get("vector_top_k", 10)
    final_top_k = retrieval_cfg.get("final_top_k", 3)
    threshold = config.get("similarity_threshold", 0.3)

    if not vault_content:
        return []

    # === BM25 search ===
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    bm25_ranked = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}

    # === Vector search ===
    vector_ranked = {}
    if vault_embeddings.nelement() > 0:
        import ollama as ollama_client
        input_embedding = ollama_client.embeddings(model=embedding_model, prompt=query)["embedding"]
        if input_embedding:
            cos_scores = torch.cosine_similarity(
                torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
            )
            # Get top vector_top_k indices
            top_k_actual = min(vector_top_k, len(cos_scores))
            top_indices = torch.topk(cos_scores, k=top_k_actual)[1].tolist()
            for rank, idx in enumerate(top_indices):
                if cos_scores[idx] >= threshold:
                    vector_ranked[idx] = rank

    # === Reciprocal Rank Fusion (RRF) ===
    # RRF score = sum(weight / (k + rank)) for each method
    rrf_k = 60  # standard RRF constant
    all_indices = set(bm25_ranked.keys()) | set(vector_ranked.keys())
    rrf_scores = {}
    for idx in all_indices:
        score = 0.0
        if idx in bm25_ranked:
            score += bm25_weight / (rrf_k + bm25_ranked[idx])
        if idx in vector_ranked:
            score += vector_weight / (rrf_k + vector_ranked[idx])
        rrf_scores[idx] = score

    # Sort by RRF score descending
    sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    top_indices = sorted_indices[:final_top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": vault_content[idx].strip(),
            "score": rrf_scores[idx],
            "index": idx,
        })
    return results


# ===== Text chunking =====

def chunk_text(text, chunk_size=300, overlap=80):
    """Split text into overlapping chunks, respecting sentence and paragraph boundaries.
    Supports both Chinese and English punctuation.
    Smaller chunks = better retrieval precision.
    """
    if not text or not text.strip():
        return []

    # Preserve paragraph structure - split by newline
    paragraphs = re.split(r'\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # First pass: build raw chunks
    raw_chunks = []

    for para in paragraphs:
        # Collapse internal whitespace but keep the paragraph as a unit
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue

        # If paragraph fits in one chunk, keep it as-is
        if len(para) <= chunk_size:
            raw_chunks.append(para)
            continue

        # Split paragraph on sentence boundaries (Chinese + English)
        sentences = re.split(r'(?<=[。！？.!?；;：:])\s*', para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # No sentence boundaries, hard split
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i:i + chunk_size]
                if chunk.strip():
                    raw_chunks.append(chunk.strip())
            continue

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (sentence + " ").strip()
            else:
                if current_chunk:
                    raw_chunks.append(current_chunk.strip())
                if len(sentence) > chunk_size:
                    for i in range(0, len(sentence), chunk_size - overlap):
                        part = sentence[i:i + chunk_size]
                        if part.strip():
                            raw_chunks.append(part.strip())
                    current_chunk = ""
                else:
                    if overlap > 0 and raw_chunks:
                        prev = raw_chunks[-1]
                        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence + " "

        if current_chunk.strip():
            raw_chunks.append(current_chunk.strip())

    # Second pass: merge short/title-only chunks with their following chunk
    # A chunk is "too short" if it looks like a title or header (< 30 chars, no sentence-ending punctuation)
    merged = []
    buffer = ""
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
        # Check if this looks like a standalone title/header
        is_short_header = (
            len(chunk) < 30
            and not re.search(r'[。！？.!?]', chunk)
            and not re.search(r'[，,；;]', chunk)
        )
        if is_short_header:
            # Merge with buffer or hold for next chunk
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
        else:
            if buffer:
                # Prepend buffered header to current chunk
                merged.append((buffer + " " + chunk).strip())
                buffer = ""
            else:
                merged.append(chunk)

    # Don't forget remaining buffer
    if buffer.strip():
        merged.append(buffer.strip())

    return merged


# ===== Context formatting =====

def format_context(chunks, registry=None):
    """Format retrieved chunks into a structured context string for the LLM.
    Includes chunk numbering and optional source attribution.
    """
    if not chunks:
        return ""

    lines = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        # Try to find source from registry
        source = ""
        if registry and isinstance(chunk, dict) and "index" in chunk:
            idx = chunk["index"]
            if idx < len(registry):
                source = registry[idx].get("source", "")
        header = f"[{i}]"
        if source:
            header += f" (来源: {source})"
        lines.append(f"{header} {text}")

    return "\n\n".join(lines)


# ===== ANSI Colors =====

class Colors:
    PINK = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


# ===== Latency Metrics =====

class MetricsTracker:
    """Lightweight latency tracker for RAG pipeline stages.
    Records per-stage timings and provides summary statistics.
    """

    STAGES = ["query_rewrite", "bm25_search", "vector_search", "rrf_fusion",
              "context_format", "llm_generate", "total"]

    def __init__(self):
        self.records = []       # list of dicts, one per query
        self._current = {}      # current query timings

    @contextmanager
    def stage(self, name):
        """Context manager to time a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._current[name] = elapsed

    def start_query(self):
        """Mark the beginning of a new query."""
        self._current = {"_ts": time.perf_counter()}

    def end_query(self):
        """Mark the end of the current query and store the record."""
        if "_ts" in self._current:
            total = time.perf_counter() - self._current.pop("_ts")
            self._current["total"] = total
            self.records.append(self._current.copy())

    def get_stats(self):
        """Return aggregated statistics across all recorded queries."""
        if not self.records:
            return {}
        stats = {}
        for stage in self.STAGES:
            vals = [r.get(stage, 0) for r in self.records if stage in r]
            if vals:
                stats[stage] = {
                    "count": len(vals),
                    "avg_ms": round(sum(vals) / len(vals) * 1000, 1),
                    "min_ms": round(min(vals) * 1000, 1),
                    "max_ms": round(max(vals) * 1000, 1),
                    "p50_ms": round(sorted(vals)[len(vals) // 2] * 1000, 1),
                }
        return stats

    def format_stats(self):
        """Format stats as a readable string."""
        stats = self.get_stats()
        if not stats:
            return "No metrics recorded."
        lines = ["=== Latency Metrics ==="]
        for stage in self.STAGES:
            if stage in stats:
                s = stats[stage]
                lines.append(
                    f"  {stage:20s}  avg={s['avg_ms']:>7.1f}ms  "
                    f"min={s['min_ms']:>7.1f}ms  max={s['max_ms']:>7.1f}ms  "
                    f"p50={s['p50_ms']:>7.1f}ms  (n={s['count']})"
                )
        return "\n".join(lines)

    def export_json(self, path):
        """Export all records to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records, "stats": self.get_stats()},
                      f, ensure_ascii=False, indent=2)
        return path
