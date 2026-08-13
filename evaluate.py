"""
evaluate.py - Retrieval quality evaluation benchmark.

Measures Recall@K and MRR (Mean Reciprocal Rank) against a ground-truth test set.

Usage:
    python evaluate.py                          # Run with default test file (test_queries.json)
    python evaluate.py --test-file my_tests.json
    python evaluate.py --generate-sample        # Generate a sample test file

Test file format (test_queries.json):
    [
      {
        "query": "What are the rules for bidding?",
        "relevant_sources": ["斗地主赛事说明.docx"]
      },
      ...
    ]

    relevant_sources: list of source filenames that contain the answer.
    If a retrieved chunk's source matches any in relevant_sources, it counts as relevant.
"""

import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import argparse

from core import (
    load_config, load_vault, load_registry,
    load_or_generate_embeddings, BM25Index,
    hybrid_search, Colors as C,
)


def load_test_cases(path):
    """Load test cases from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sample(path):
    """Generate a sample test file."""
    sample = [
        {
            "query": "这里填写你的测试问题",
            "relevant_sources": ["填写相关的文档文件名.docx"],
        },
        {
            "query": "第二个测试问题",
            "relevant_sources": ["另一个文档.pdf"],
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"{C.GREEN}✓ Sample test file generated: {path}{C.RESET}")
    print(f"  Edit it with your actual queries and expected sources.")


def evaluate_query(query, relevant_sources, vault_embeddings, vault_content,
                   bm25_index, embedding_model, config, registry):
    """Evaluate retrieval for a single query.
    Returns dict with retrieved sources and whether each is relevant.
    """
    results = hybrid_search(
        query, vault_embeddings, vault_content,
        bm25_index, embedding_model, config
    )

    retrieved = []
    for r in results:
        idx = r["index"]
        source = registry[idx].get("source", "unknown") if idx < len(registry) else "unknown"
        is_relevant = source in relevant_sources
        retrieved.append({
            "source": source,
            "score": r["score"],
            "relevant": is_relevant,
        })

    return retrieved


def compute_metrics(results, k_values=[1, 3, 5]):
    """Compute Recall@K and MRR from evaluation results.
    results: list of list of {"relevant": bool, ...}
    """
    metrics = {}

    # MRR (Mean Reciprocal Rank)
    rr_sum = 0.0
    for retrieved in results:
        for rank, r in enumerate(retrieved, 1):
            if r["relevant"]:
                rr_sum += 1.0 / rank
                break
    metrics["mrr"] = rr_sum / len(results) if results else 0.0

    # Recall@K
    for k in k_values:
        recall_sum = 0.0
        for retrieved in results:
            top_k = retrieved[:k]
            # How many relevant docs in top-K?
            relevant_in_k = sum(1 for r in top_k if r["relevant"])
            # Total relevant for this query (from ground truth)
            # We count relevant in ALL retrieved as proxy
            total_relevant = sum(1 for r in retrieved if r["relevant"])
            if total_relevant > 0:
                recall_sum += min(relevant_in_k / total_relevant, 1.0)
        metrics[f"recall@{k}"] = recall_sum / len(results) if results else 0.0

    return metrics


def run_evaluation(test_file, config):
    """Run full evaluation."""
    # Load resources
    vault_file = config["vault_file"]
    registry_file = config["registry_file"]
    cache_file = config["embeddings_cache_file"]
    embedding_model = config["embedding_model"]

    print(f"{C.GREEN}Loading vault and embeddings...{C.RESET}")
    vault_embeddings, vault_content = load_or_generate_embeddings(
        vault_file, embedding_model, cache_file=cache_file
    )
    bm25_index = BM25Index(vault_content)
    registry = load_registry(registry_file)

    print(f"{C.GREEN}Loaded {len(vault_content)} chunks from "
          f"{len(set(e.get('source', '') for e in registry))} documents.{C.RESET}\n")

    # Load test cases
    test_cases = load_test_cases(test_file)
    print(f"{C.GREEN}Running {len(test_cases)} test queries...{C.RESET}\n")

    all_results = []
    details = []

    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        relevant_sources = set(tc["relevant_sources"])

        retrieved = evaluate_query(
            query, relevant_sources, vault_embeddings, vault_content,
            bm25_index, embedding_model, config, registry
        )
        all_results.append(retrieved)

        # Per-query detail
        hits = sum(1 for r in retrieved if r["relevant"])
        status = f"{C.GREEN}✓{C.RESET}" if hits > 0 else f"{C.RED}✗{C.RESET}"
        print(f"  {status} [{i}/{len(test_cases)}] {query[:60]}")
        print(f"       Hits: {hits}/{len(retrieved)} | Sources: "
              f"{', '.join(r['source'] for r in retrieved[:3])}")

        details.append({
            "query": query,
            "expected": list(relevant_sources),
            "retrieved": [{"source": r["source"], "score": round(r["score"], 4),
                           "relevant": r["relevant"]} for r in retrieved],
            "hits": hits,
        })

    # Compute aggregate metrics
    retrieval_cfg = config.get("retrieval", {})
    final_k = retrieval_cfg.get("final_top_k", 3)
    k_values = [k for k in [1, 3, 5, 10] if k <= max(final_k, 5)]

    metrics = compute_metrics(all_results, k_values=k_values)

    print(f"\n{'='*50}")
    print(f"{C.CYAN}=== Evaluation Results ==={C.RESET}")
    print(f"  Queries:      {len(test_cases)}")
    print(f"  Vault chunks: {len(vault_content)}")
    print(f"  Documents:    {len(set(e.get('source', '') for e in registry))}")
    print()
    print(f"  {C.YELLOW}MRR:          {metrics['mrr']:.3f}{C.RESET}")
    for k in k_values:
        key = f"recall@{k}"
        val = metrics[key]
        color = C.GREEN if val >= 0.7 else C.YELLOW if val >= 0.4 else C.RED
        print(f"  {color}{key:14s} {val:.3f}{C.RESET}")
    print(f"{'='*50}")

    return {"metrics": metrics, "details": details}


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--test-file", default="test_queries.json",
                        help="Path to test queries JSON (default: test_queries.json)")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a sample test file and exit")
    parser.add_argument("--output", metavar="FILE",
                        help="Export detailed results to JSON")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample(args.test_file)
        return

    if not os.path.exists(args.test_file):
        print(f"{C.RED}Test file not found: {args.test_file}{C.RESET}")
        print(f"Run with --generate-sample to create a template.")
        return

    config = load_config()
    results = run_evaluation(args.test_file, config)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n{C.GREEN}✓ Results exported to: {args.output}{C.RESET}")


if __name__ == "__main__":
    main()
