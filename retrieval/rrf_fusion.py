"""
retrieval/rrf_fusion.py
Reciprocal Rank Fusion (RRF) — merges ranked result lists from multiple retrievers.

RRF is the standard fusion technique for hybrid search. It works by assigning each
document a score based on its rank position in each list, then combining those scores.

Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
    where k=60 (standard constant), rank_i = position in retriever i's results

Why RRF over simple score normalization?
- Scores from different retrievers (BM25 vs cosine) are NOT comparable
- RRF only uses rank positions, making it score-agnostic
- Proven to outperform linear combination in academic benchmarks
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Standard RRF constant — higher k reduces the impact of top-rank differences
RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: List[List[Dict[str, Any]]],
    top_k: int = 20,
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result dict MUST have a 'text' field (used as the document identity key).

    Args:
        *result_lists:  Variable number of ranked result lists.
                        Each list should be sorted by relevance (best first).
        top_k:          Number of results to return after fusion.
        k:              RRF constant (default 60).

    Returns:
        Merged and re-ranked list of results, sorted by RRF score descending.
        Each result has an added 'rrf_score' field.
    """
    # Map: document_text → {rrf_score, best_doc_dict, sources}
    doc_scores: Dict[str, Dict[str, Any]] = {}

    for list_idx, results in enumerate(result_lists):
        if not results:
            continue

        for rank, doc in enumerate(results, start=1):
            text = doc.get("text", "").strip()
            if not text:
                continue

            # Use first 200 chars as identity key (full text match is too expensive)
            doc_key = text[:200]

            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "rrf_score": 0.0,
                    "doc": doc,
                    "sources": [],
                }

            # RRF formula: 1 / (k + rank)
            doc_scores[doc_key]["rrf_score"] += 1.0 / (k + rank)
            source_label = doc.get("retrieval_method", f"retriever_{list_idx}")
            doc_scores[doc_key]["sources"].append(source_label)

    # Sort by RRF score descending
    ranked = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Build final results
    results = []
    for item in ranked[:top_k]:
        doc = item["doc"].copy()
        doc["rrf_score"] = round(item["rrf_score"], 6)
        doc["retrieval_sources"] = list(set(item["sources"]))
        # Keep the original score for reference
        if "score" not in doc:
            doc["score"] = doc["rrf_score"]
        results.append(doc)

    logger.info(
        "RRF: Merged %d lists → %d unique docs → returning top %d",
        len(result_lists),
        len(doc_scores),
        len(results),
    )
    return results
