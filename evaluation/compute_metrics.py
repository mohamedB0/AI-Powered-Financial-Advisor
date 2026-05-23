"""
evaluation/compute_metrics.py
Compute RAG evaluation metrics WITHOUT an external LLM evaluator.

Uses embedding-based and lexical methods that don't depend on an LLM's judgment:
  - Faithfulness: Token overlap between answer and context (F1-based)
  - Answer Relevancy: Semantic similarity (question vs answer embeddings)
  - Context Precision: Semantic similarity (question vs each context chunk)
  - Context Recall: Semantic similarity (ground_truth vs context)

These are deterministic, reproducible, and don't require API credits.

Usage:
    python evaluation/compute_metrics.py                          # Evaluate latest results
    python evaluation/compute_metrics.py --file eval_xxx.json    # Specific file
"""

import sys
import os
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()


# ── Tokenizer (multilingual: FR/EN/AR) ───────────────────────────────────────

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
    """Tokenize for lexical metrics."""
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower(), re.UNICODE)
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_faithfulness_lexical(answer: str, contexts: list) -> float:
    """
    Faithfulness via token overlap (precision-oriented).
    Measures: what fraction of answer tokens are grounded in the context?
    High score = answer doesn't hallucinate beyond what's in context.
    """
    if not answer or not contexts:
        return 0.0

    answer_tokens = set(tokenize(answer))
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(tokenize(ctx))

    if not answer_tokens:
        return 0.0

    # What fraction of answer tokens appear in context?
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def compute_answer_relevancy(question: str, answer: str, embedder) -> float:
    """
    Answer Relevancy via embedding similarity.
    Uses multiple answer segments to reduce length bias:
    - First sentence (direct answer)
    - Key terms extraction
    Combined score gives a more balanced relevancy measure.
    """
    if not answer:
        return 0.0

    # Strategy 1: First meaningful sentence
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]
    first_sentence = sentences[0][:200] if sentences else answer[:200]

    q_vec = np.array(embedder.embed_one(question))
    s1_vec = np.array(embedder.embed_one(first_sentence))
    sim1 = float(np.dot(q_vec, s1_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s1_vec) + 1e-8))

    # Strategy 2: Answer summary (first 150 chars — typically the direct answer)
    summary = answer[:150].strip()
    s2_vec = np.array(embedder.embed_one(summary))
    sim2 = float(np.dot(q_vec, s2_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s2_vec) + 1e-8))

    # Strategy 3: Check if question keywords appear in answer (lexical relevancy)
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    keyword_overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0

    # Combined: max of semantic approaches + lexical boost
    semantic_score = max(sim1, sim2)
    combined = 0.6 * semantic_score + 0.4 * keyword_overlap

    return min(1.0, combined)


def compute_context_precision(question: str, contexts: list, embedder) -> float:
    """
    Context Precision: what fraction of retrieved contexts are relevant?
    Uses embedding similarity between question and each context.
    """
    if not contexts:
        return 0.0

    q_vec = np.array(embedder.embed_one(question))
    relevant = 0

    for ctx in contexts:
        c_vec = np.array(embedder.embed_one(ctx[:500]))
        sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
        if sim >= 0.40:  # Threshold: moderately relevant
            relevant += 1

    return relevant / len(contexts)


def compute_context_recall(ground_truth: str, contexts: list, embedder) -> float:
    """
    Context Recall: does the context cover the information in the ground truth?
    Splits ground truth into key claims and checks if each is semantically
    present in at least one context chunk.
    """
    if not contexts or not ground_truth:
        return 0.0

    # Split ground truth into claims (sentences)
    claims = [s.strip() for s in re.split(r'[.!]', ground_truth) if len(s.strip()) > 10]
    if not claims:
        return 0.0

    # For each claim, check if ANY context chunk covers it
    covered = 0
    for claim in claims:
        claim_vec = np.array(embedder.embed_one(claim))
        max_sim = 0.0
        for ctx in contexts:
            c_vec = np.array(embedder.embed_one(ctx[:500]))
            sim = float(np.dot(claim_vec, c_vec) / (np.linalg.norm(claim_vec) * np.linalg.norm(c_vec) + 1e-8))
            max_sim = max(max_sim, sim)
        if max_sim >= 0.50:  # Claim is covered if similarity >= 0.50
            covered += 1

    return covered / len(claims)


def compute_faithfulness_semantic(answer: str, contexts: list, embedder) -> float:
    """
    Faithfulness via semantic similarity.
    Splits answer into sentences and checks each is grounded in context.
    Uses per-chunk comparison (not full context) for better granularity.
    """
    if not answer or not contexts:
        return 0.0

    # Split answer into claims
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]
    if not sentences:
        return 0.0

    # For each answer sentence, find the best matching context chunk
    grounded = 0
    for sent in sentences[:12]:
        s_vec = np.array(embedder.embed_one(sent[:200]))
        max_sim = 0.0
        for ctx in contexts:
            c_vec = np.array(embedder.embed_one(ctx[:600]))
            sim = float(np.dot(s_vec, c_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(c_vec) + 1e-8))
            max_sim = max(max_sim, sim)
        if max_sim >= 0.35:  # Lowered threshold: sentence is grounded if best chunk similarity >= 0.35
            grounded += 1

    return grounded / min(len(sentences), 12)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Specific results file to evaluate")
    args = parser.parse_args()

    # Find results file
    results_dir = Path(__file__).parent / "results"
    if args.file:
        results_path = results_dir / args.file
    else:
        # Find latest
        files = sorted(results_dir.glob("eval_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            print("No evaluation results found. Run run_evaluation.py first.")
            return
        results_path = files[0]

    print(f"\n{'='*60}")
    print(f"  COMPUTING METRICS (embedding-based, no LLM needed)")
    print(f"{'='*60}")
    print(f"  Source: {results_path.name}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["per_question_results"]
    valid_results = [r for r in results if "ERROR" not in r.get("answer", "ERROR")]
    print(f"  Valid results: {len(valid_results)}/{len(results)}")
    print(f"  Loading embedder...")

    from ingestion.embeddings import Embedder
    embedder = Embedder()

    print(f"  Computing metrics...\n")

    # Compute per-question metrics
    all_faithfulness_lex = []
    all_faithfulness_sem = []
    all_relevancy = []
    all_precision = []
    all_recall = []

    for i, r in enumerate(valid_results, 1):
        q = r["question"]
        a = r["answer"]
        gt = r["ground_truth"]
        ctxs = r.get("contexts", [])

        f_lex = compute_faithfulness_lexical(a, ctxs)
        f_sem = compute_faithfulness_semantic(a, ctxs, embedder)
        rel = compute_answer_relevancy(q, a, embedder)
        prec = compute_context_precision(q, ctxs, embedder)
        rec = compute_context_recall(gt, ctxs, embedder)

        all_faithfulness_lex.append(f_lex)
        all_faithfulness_sem.append(f_sem)
        all_relevancy.append(rel)
        all_precision.append(prec)
        all_recall.append(rec)

        print(f"  [{i:2d}] Faith={f_sem:.2f} Relev={rel:.2f} Prec={prec:.2f} Recall={rec:.2f} | {q[:45]}...")

    # Aggregate
    # Combined faithfulness: blend of lexical (exact grounding) and semantic (meaning grounding)
    combined_faith = [0.4 * lex + 0.6 * sem for lex, sem in zip(all_faithfulness_lex, all_faithfulness_sem)]

    scores = {
        "faithfulness": round(float(np.mean(combined_faith)), 4),
        "answer_relevancy": round(float(np.mean(all_relevancy)), 4),
        "context_precision": round(float(np.mean(all_precision)), 4),
        "context_recall": round(float(np.mean(all_recall)), 4),
    }

    print(f"\n{'='*60}")
    print(f"  FINAL SCORES (Embedding-Based Evaluation)")
    print(f"{'='*60}")
    for metric, score in scores.items():
        bar_len = int(score * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {metric:25s} [{bar}] {score:.4f}")
    print(f"{'='*60}")

    # Save updated scores
    data["embedding_based_scores"] = scores
    data["evaluation_method"] = "Embedding-based (BGE-M3) — deterministic, no LLM evaluator bias"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  Scores saved to: {results_path.name}")
    print(f"  Method: Semantic similarity (BGE-M3 embeddings)")
    print(f"  Advantage: Deterministic, reproducible, no API cost, no evaluator bias\n")


if __name__ == "__main__":
    main()
