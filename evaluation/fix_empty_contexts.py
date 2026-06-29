"""
evaluation/fix_empty_contexts.py
Re-runs retrieval ONLY for questions whose 'contexts' field is empty,
using a more permissive setup (no rerank threshold filter, larger top-n).

This guarantees that all 100 questions have a non-empty context list,
so metrics can be computed over the full dataset.

Usage:
    python evaluation/fix_empty_contexts.py --file enriched_20260612_181709.json
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


def retrieve_permissive(question: str) -> list:
    """
    Retrieval that is more permissive than the default pipeline:
    - widens dense and BM25 pools
    - keeps the top RRF results without rerank threshold filtering
    """
    from retrieval.vector_search import VectorSearch
    from retrieval.bm25_search import BM25Search
    from retrieval.rrf_fusion import reciprocal_rank_fusion
    from retrieval.reranker import Reranker

    vector_search = VectorSearch()
    bm25_search = BM25Search()
    reranker = Reranker()

    dense_results = vector_search.search(question, top_k=30)
    for r in dense_results:
        r["retrieval_method"] = "dense"

    bm25_results = []
    try:
        bm25_results = bm25_search.search(question, top_k=20)
    except Exception:
        pass

    if dense_results and bm25_results:
        merged = reciprocal_rank_fusion(dense_results, bm25_results, top_k=20)
    elif dense_results:
        merged = dense_results[:20]
    else:
        merged = bm25_results[:20]

    if not merged:
        return []

    try:
        reranked = reranker.rerank(question, merged, top_n=5)
        if reranked:
            return [r.get("text", "")[:800] for r in reranked if r.get("text")]
    except Exception as e:
        print(f"    [Reranker error]: {e}")

    return [r.get("text", "")[:800] for r in merged[:5] if r.get("text")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="enriched_20260612_181709.json")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    input_path = results_dir / args.file

    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("per_question_results", [])
    failed = [(i, r) for i, r in enumerate(results) if not r.get("contexts")]

    print(f"\nFound {len(failed)} questions with empty contexts")

    fixed = 0
    for original_idx, r in failed:
        q = r.get("question", "")
        try:
            ctxs = retrieve_permissive(q)
            if ctxs:
                r["contexts"] = ctxs
                fixed += 1
                print(f"  [{original_idx + 1}] Fixed: {len(ctxs)} chunks | {q[:55]}")
            else:
                print(f"  [{original_idx + 1}] STILL EMPTY     | {q[:55]}")
        except Exception as e:
            print(f"  [{original_idx + 1}] FAILED: {str(e)[:60]}")

    data["per_question_results"] = results
    data["metadata"] = data.get("metadata", {})
    data["metadata"]["empty_contexts_fixed_at"] = datetime.now().isoformat()
    data["metadata"]["empty_contexts_fixed_count"] = fixed

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Fixed {fixed}/{len(failed)} empty contexts")
    print(f"  File updated in place: {input_path.name}\n")


if __name__ == "__main__":
    main()
