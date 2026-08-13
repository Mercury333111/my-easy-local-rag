"""
localrag.py - Local RAG chat with Ollama (Hybrid Retrieval).

Usage:
    python localrag.py                          # Start chat with query rewrite
    python localrag.py --no-rewrite             # Start chat without query rewrite
    python localrag.py --model qwen3.6          # Specify model
    python localrag.py --export chat.json       # Export chat after session ends
    python localrag.py --rebuild-cache          # Force regenerate embeddings
    python localrag.py --metrics                # Show latency metrics after session
    python localrag.py --metrics metrics.json   # Export metrics to JSON
"""

import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import argparse
from datetime import datetime

import torch
from openai import OpenAI

from core import (
    load_config, load_vault, load_registry,
    load_or_generate_embeddings, BM25Index,
    hybrid_search, format_context, Colors as C,
    MetricsTracker,
)


# ===== Query Rewrite =====

def rewrite_query(user_input, conversation_history, client, ollama_model):
    """Rewrite the user query using conversation context for better retrieval."""
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
The rewritten query should:

- Preserve the core intent and meaning of the original query
- Expand and clarify the query to make it more specific and informative for retrieving relevant context
- Avoid introducing new topics or queries that deviate from the original query
- DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

Return ONLY the rewritten query text, without any additional formatting or explanations.

Conversation History:
{context}

Original query: [{user_input}]

Rewritten query:
"""
    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            n=1,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        return result if result else user_input
    except Exception as e:
        print(f"{C.YELLOW}[Query rewrite failed: {e}]{C.RESET}")
        return user_input


# ===== Chat =====

def chat(user_input, system_message, vault_embeddings, vault_content,
         bm25_index, registry, config, client, ollama_model, embedding_model,
         conversation_history, use_rewrite=True, metrics=None):
    """Process one turn of conversation using hybrid retrieval."""
    if metrics:
        metrics.start_query()

    conversation_history.append({"role": "user", "content": user_input})

    # Query rewrite (only after first turn)
    if use_rewrite and len(conversation_history) > 1:
        with metrics.stage("query_rewrite") if metrics else _noop_ctx():
            rewritten = rewrite_query(user_input, conversation_history, client, ollama_model)
        print(f"{C.PINK}Original:  {user_input}{C.RESET}")
        print(f"{C.PINK}Rewritten: {rewritten}{C.RESET}")
        query = rewritten
    else:
        query = user_input

    # Hybrid retrieval (timed internally for sub-stages)
    retrieval_cfg = config.get("retrieval", {})
    bm25_top_k = retrieval_cfg.get("bm25_top_k", 10)
    vector_top_k = retrieval_cfg.get("vector_top_k", 10)
    final_top_k = retrieval_cfg.get("final_top_k", 3)

    # BM25 search
    with metrics.stage("bm25_search") if metrics else _noop_ctx():
        bm25_results = bm25_index.search(query, top_k=bm25_top_k)

    # Vector search
    with metrics.stage("vector_search") if metrics else _noop_ctx():
        import ollama as ollama_client
        vector_ranked = {}
        if vault_embeddings.nelement() > 0:
            input_embedding = ollama_client.embeddings(model=embedding_model, prompt=query)["embedding"]
            if input_embedding:
                cos_scores = torch.cosine_similarity(
                    torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
                )
                top_k_actual = min(vector_top_k, len(cos_scores))
                top_indices = torch.topk(cos_scores, k=top_k_actual)[1].tolist()
                threshold = config.get("similarity_threshold", 0.3)
                for rank, idx in enumerate(top_indices):
                    if cos_scores[idx] >= threshold:
                        vector_ranked[idx] = rank

    # RRF fusion
    with metrics.stage("rrf_fusion") if metrics else _noop_ctx():
        bm25_ranked = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
        rrf_k = 60
        bm25_weight = retrieval_cfg.get("bm25_weight", 0.4)
        vector_weight = retrieval_cfg.get("vector_weight", 0.6)
        all_indices = set(bm25_ranked.keys()) | set(vector_ranked.keys())
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in bm25_ranked:
                score += bm25_weight / (rrf_k + bm25_ranked[idx])
            if idx in vector_ranked:
                score += vector_weight / (rrf_k + vector_ranked[idx])
            rrf_scores[idx] = score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:final_top_k]
        chunks = [{"text": vault_content[i].strip(), "score": rrf_scores[i], "index": i}
                  for i in top_indices]

    # Context formatting
    with metrics.stage("context_format") if metrics else _noop_ctx():
        if chunks:
            context_str = format_context(chunks, registry)
            max_context_chars = 4000
            if len(context_str) > max_context_chars:
                context_str = context_str[:max_context_chars] + "..."
            print(f"\n{C.CYAN}[Retrieved {len(chunks)} chunks]:\n{context_str}{C.RESET}\n")
        else:
            context_str = ""
            print(f"{C.CYAN}[No relevant context found]{C.RESET}\n")

    # Build user message with structured context
    user_msg = user_input
    if context_str:
        user_msg = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请如实说明。

参考资料：
{context_str}

问题：{user_input}"""
    conversation_history[-1]["content"] = user_msg

    # Generate response
    with metrics.stage("llm_generate") if metrics else _noop_ctx():
        try:
            messages = [{"role": "system", "content": system_message}, *conversation_history]
            response = client.chat.completions.create(
                model=ollama_model,
                messages=messages,
                max_tokens=2000,
            )
            assistant_msg = response.choices[0].message.content
            if not assistant_msg or not assistant_msg.strip():
                assistant_msg = "[Model returned empty response. Try rephrasing your question.]"
        except Exception as e:
            assistant_msg = f"[Error calling model: {e}]"

    conversation_history.append({"role": "assistant", "content": assistant_msg})

    if metrics:
        metrics.end_query()

    return assistant_msg


def _noop_ctx():
    """No-op context manager when metrics is disabled."""
    import contextlib
    return contextlib.nullcontext()


# ===== Export =====

def export_chat(conversation_history, export_path, format="json"):
    """Export chat history to file."""
    if not conversation_history:
        print(f"{C.YELLOW}No conversation to export.{C.RESET}")
        return

    if format == "json":
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "messages": conversation_history,
            }, f, ensure_ascii=False, indent=2)

    elif format == "markdown":
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"# RAG Chat Export\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
            for msg in conversation_history:
                role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
                content = msg["content"]
                # Strip context section for cleaner export
                if msg["role"] == "user" and "参考资料：" in content and "问题：" in content:
                    content = content.split("问题：")[-1].strip()
                f.write(f"### {role}\n\n{content}\n\n---\n\n")

    print(f"{C.GREEN}✓ Chat exported to: {export_path}{C.RESET}")


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="Local RAG Chat with Ollama")
    parser.add_argument("--model", default=None, help="Ollama model (default: from config)")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable query rewrite")
    parser.add_argument("--export", metavar="FILE", help="Export chat to file (.json or .md)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force regenerate embeddings")
    parser.add_argument("--metrics", metavar="FILE", nargs="?", const="__print__",
                        help="Enable latency metrics. Optionally export to JSON file.")
    args = parser.parse_args()

    # Load config
    config = load_config()
    ollama_model = args.model or config["ollama_model"]
    embedding_model = config["embedding_model"]
    system_message = config["system_message"]
    vault_file = config["vault_file"]
    registry_file = config["registry_file"]
    cache_file = config["embeddings_cache_file"]

    # Init API client
    api_cfg = config["ollama_api"]
    client = OpenAI(base_url=api_cfg["base_url"], api_key=api_cfg.get("api_key", ""))

    # Load vault content
    print(f"{C.GREEN}Loading vault...{C.RESET}")
    vault_content = load_vault(vault_file)
    registry = load_registry(registry_file)

    # Load embeddings (with cache)
    print(f"{C.GREEN}Loading embeddings...{C.RESET}")
    vault_embeddings, vault_content = load_or_generate_embeddings(
        vault_file, embedding_model, cache_file=cache_file, force=args.rebuild_cache
    )

    # Build BM25 index
    print(f"{C.GREEN}Building BM25 index...{C.RESET}")
    bm25_index = BM25Index(vault_content)

    retrieval_cfg = config.get("retrieval", {})
    print(f"{C.GREEN}Ready. {len(vault_content)} chunks loaded.{C.RESET}")
    print(f"{C.GREEN}Model: {ollama_model} | Rewrite: {'ON' if not args.no_rewrite else 'OFF'}{C.RESET}")
    print(f"{C.GREEN}Retrieval: BM25({retrieval_cfg.get('bm25_weight', 0.4)}) + "
          f"Vector({retrieval_cfg.get('vector_weight', 0.6)}) | "
          f"Top-K: {retrieval_cfg.get('final_top_k', 3)}{C.RESET}\n")

    # Metrics tracker
    metrics = MetricsTracker() if args.metrics else None

    # Chat loop
    conversation_history = []
    while True:
        try:
            user_input = input(f"{C.YELLOW}>>> {C.RESET}")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ('quit', 'exit', 'q'):
            break
        if not user_input.strip():
            continue

        response = chat(
            user_input, system_message, vault_embeddings, vault_content,
            bm25_index, registry, config, client, ollama_model, embedding_model,
            conversation_history, use_rewrite=not args.no_rewrite,
            metrics=metrics,
        )
        print(f"{C.GREEN}{response}{C.RESET}\n")

    # Export if requested
    if args.export:
        ext = os.path.splitext(args.export)[1].lower()
        fmt = "markdown" if ext in (".md", ".markdown") else "json"
        export_chat(conversation_history, args.export, format=fmt)

    # Display metrics
    if metrics and metrics.records:
        print(f"\n{metrics.format_stats()}")
        if args.metrics and args.metrics != "__print__":
            path = metrics.export_json(args.metrics)
            print(f"{C.GREEN}✓ Metrics exported to: {path}{C.RESET}")

    print(f"\n{C.GREEN}Goodbye!{C.RESET}")


if __name__ == "__main__":
    main()
