"""
evaluation/enrich_with_contexts.py
Re-runs the retrieval pipeline ONLY (no LLM generation) on an existing
evaluation file to add the `contexts` field required by compute_metrics.py.

The existing answers and ground truths are preserved untouched.

Usage:
    python evaluation/enrich_with_contexts.py
    python evaluation/enrich_with_contexts.py --input judge_eval_20260604_144416.json
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()


def retrieve_only(question: str) -> list:
    """Run the hybrid retrieval (no LLM call) and return the context list."""
    from retrieval.vector_search import VectorSearch
    from retrieval.bm25_search import BM25Search
    from retrieval.rrf_fusion import reciprocal_rank_fusion
    from retrieval.reranker import Reranker

    vector_search = VectorSearch()
    bm25_search = BM25Search()
    reranker = Reranker()

    dense_results = vector_search.search(question, top_k=20)
    for r in dense_results:
        r["retrieval_method"] = "dense"

    bm25_results = []
    try:
        bm25_results = bm25_search.search(question, top_k=15)
    except Exception:
        pass

    if dense_results and bm25_results:
        merged = reciprocal_rank_fusion(dense_results, bm25_results, top_k=15)
    elif dense_results:
        merged = dense_results[:15]
    else:
        merged = bm25_results[:15]

    if merged:
        try:
            merged = reranker.rerank(question, merged, top_n=5)
            merged = [r for r in merged if r.get("rerank_score", 0) >= 0.2]
        except Exception as e:
            print(f"    [Reranker error, using top 5 RRF instead]: {e}")
            merged = merged[:5]

    return [r.get("text", "")[:800] for r in merged if r.get("text")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="judge_eval_20260604_144416.json",
                        help="Filename inside evaluation/results/")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (defaults to enriched_<timestamp>.json)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    input_path = results_dir / args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ENRICHING EVALUATION FILE WITH RETRIEVAL CONTEXTS")
    print(f"{'='*60}")
    print(f"  Input: {input_path.name}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("per_question_results", [])
    print(f"  Questions to enrich: {len(results)}")
    print(f"  Loading retrieval components...\n")

    enriched = []
    for i, r in enumerate(results, 1):
        q = r.get("question", "")
        try:
            t0 = time.time()
            ctxs = retrieve_only(q)
            elapsed = (time.time() - t0) * 1000
            r_new = dict(r)
            r_new["contexts"] = ctxs
            enriched.append(r_new)
            print(f"  [{i:3d}/{len(results)}] {len(ctxs)} chunks  {elapsed:6.0f}ms  | {q[:55]}")
        except Exception as e:
            print(f"  [{i:3d}/{len(results)}] FAILED: {str(e)[:70]}")
            r_new = dict(r)
            r_new["contexts"] = []
            enriched.append(r_new)

    data["per_question_results"] = enriched
    data["metadata"] = data.get("metadata", {})
    data["metadata"]["contexts_enriched_at"] = datetime.now().isoformat()
    data["metadata"]["retrieval_for_metrics"] = "Dense (BGE-M3, top_k=20) + BM25 (top_k=15) + RRF (top_k=15) + Jina Reranker (top_n=5)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.output or f"enriched_{timestamp}.json"
    out_path = results_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Enriched file saved to: {out_path.name}")
    print(f"  You can now run:")
    print(f"      python evaluation/compute_metrics.py --file {out_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
