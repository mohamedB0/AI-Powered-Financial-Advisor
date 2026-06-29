"""
evaluation/compute_metrics_fast.py
Fast version of compute_metrics.py.

Same metrics, but uses batch embedding to avoid hundreds of single calls per
question. Each question processes its question + answer + sentences + contexts
in one or two batch embed calls.

Usage:
    python evaluation/compute_metrics_fast.py --file enriched_xxx.json
"""

import sys
import os
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()


# ── Tokenizer ────────────────────────────────────────────────────────────────

STOPWORDS = frozenset([
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "en",
    "dans", "sur", "pour", "par", "avec", "que", "qui", "est", "sont",
    "ce", "cette", "ces", "pas", "plus", "ne", "se", "au", "aux",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "that", "this", "it", "not",
])


def tokenize(text: str) -> list:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower(), re.UNICODE)
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Per-question metric block ────────────────────────────────────────────────

def evaluate_one(question, answer, ground_truth, contexts, embedder):
    """All metrics for one question, with batched embeddings."""

    # ---- Lexical faithfulness (no embedding needed) ----
    answer_tokens = set(tokenize(answer))
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(tokenize(ctx))
    f_lex = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0

    # ---- Build all texts to embed in one batch ----
    answer_sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15][:12]
    truth_sentences = [s.strip() for s in re.split(r'[.!]', ground_truth) if len(s.strip()) > 10]
    first_sentence = answer_sentences[0][:200] if answer_sentences else answer[:200]
    summary = answer[:150].strip()
    answer_for_relev = answer[:500]

    truncated_contexts = [c[:600] for c in contexts]

    texts = [question, first_sentence, summary, answer_for_relev]
    texts.extend(s[:200] for s in answer_sentences)
    texts.extend(truth_sentences)
    texts.extend(truncated_contexts)

    if not texts:
        return {"faith_lex": f_lex, "faith_sem": 0.0, "rel": 0.0, "prec": 0.0, "rec": 0.0}

    vecs = np.array(embedder.embed(texts))

    idx = 0
    q_vec = vecs[idx]; idx += 1
    s1_vec = vecs[idx]; idx += 1
    sum_vec = vecs[idx]; idx += 1
    arelev_vec = vecs[idx]; idx += 1
    sent_vecs = vecs[idx:idx + len(answer_sentences)]; idx += len(answer_sentences)
    truth_vecs = vecs[idx:idx + len(truth_sentences)]; idx += len(truth_sentences)
    ctx_vecs = vecs[idx:idx + len(truncated_contexts)]; idx += len(truncated_contexts)

    # ---- Answer Relevancy ----
    sim1 = cosine(q_vec, s1_vec)
    sim2 = cosine(q_vec, sum_vec)
    semantic = max(sim1, sim2)
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    keyword_overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
    rel = min(1.0, 0.6 * semantic + 0.4 * keyword_overlap)

    # ---- Context Precision ----
    if len(ctx_vecs) > 0:
        relevant = sum(1 for c in ctx_vecs if cosine(q_vec, c) >= 0.40)
        prec = relevant / len(ctx_vecs)
    else:
        prec = 0.0

    # ---- Context Recall ----
    if len(truth_vecs) > 0 and len(ctx_vecs) > 0:
        covered = 0
        for tv in truth_vecs:
            max_sim = max(cosine(tv, c) for c in ctx_vecs)
            if max_sim >= 0.50:
                covered += 1
        rec = covered / len(truth_vecs)
    else:
        rec = 0.0

    # ---- Faithfulness Semantic ----
    if len(sent_vecs) > 0 and len(ctx_vecs) > 0:
        grounded = 0
        for sv in sent_vecs:
            max_sim = max(cosine(sv, c) for c in ctx_vecs)
            if max_sim >= 0.35:
                grounded += 1
        f_sem = grounded / len(sent_vecs)
    else:
        f_sem = 0.0

    return {"faith_lex": f_lex, "faith_sem": f_sem, "rel": rel, "prec": prec, "rec": rec}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_path = results_dir / args.file

    print(f"\n{'='*60}")
    print(f"  COMPUTING METRICS (fast batched, no LLM needed)")
    print(f"{'='*60}")
    print(f"  Source: {results_path.name}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["per_question_results"]
    valid = [r for r in results if "ERROR" not in r.get("answer", "ERROR") and r.get("contexts")]
    print(f"  Valid: {len(valid)}/{len(results)}")
    print(f"  Loading embedder...")

    from ingestion.embeddings import Embedder
    embedder = Embedder()

    print(f"  Computing metrics...\n")

    all_lex, all_sem, all_rel, all_prec, all_rec = [], [], [], [], []

    import time
    for i, r in enumerate(valid, 1):
        q = r["question"]
        a = r["answer"]
        gt = r["ground_truth"]
        ctxs = r.get("contexts", [])

        t0 = time.time()
        m = evaluate_one(q, a, gt, ctxs, embedder)
        dt = (time.time() - t0) * 1000

        all_lex.append(m["faith_lex"])
        all_sem.append(m["faith_sem"])
        all_rel.append(m["rel"])
        all_prec.append(m["prec"])
        all_rec.append(m["rec"])

        print(f"  [{i:3d}/{len(valid)}] F={m['faith_sem']:.2f} "
              f"R={m['rel']:.2f} P={m['prec']:.2f} Rec={m['rec']:.2f} "
              f"({dt:5.0f}ms) | {q[:42]}...")

    combined_faith = [0.4 * lx + 0.6 * sm for lx, sm in zip(all_lex, all_sem)]

    scores = {
        "faithfulness": round(float(np.mean(combined_faith)), 4),
        "answer_relevancy": round(float(np.mean(all_rel)), 4),
        "context_precision": round(float(np.mean(all_prec)), 4),
        "context_recall": round(float(np.mean(all_rec)), 4),
    }

    print(f"\n{'='*60}")
    print(f"  FINAL SCORES (Embedding-Based, RAGAS-style)")
    print(f"{'='*60}")
    for metric, score in scores.items():
        bar_len = int(score * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {metric:25s} [{bar}] {score:.4f}")
    print(f"{'='*60}")

    data["embedding_based_scores"] = scores
    data["evaluation_method"] = "Embedding-based (BGE-M3) — deterministic, no LLM evaluator"
    data["metadata"] = data.get("metadata", {})
    data["metadata"]["embedding_metrics_at"] = datetime.now().isoformat()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Scores saved to: {results_path.name}")
    print(f"  All metrics: {scores}\n")


if __name__ == "__main__":
    main()
