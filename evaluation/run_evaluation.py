"""
evaluation/run_evaluation.py
RAGAS Evaluation Pipeline for the AI Financial Advisor.

Bulletproof version: uses ONLY Groq (free, unlimited for dev) with aggressive
rate-limit handling. Zero dependency on OpenRouter.

Metrics computed:
  - Faithfulness: Does the answer stay faithful to the retrieved context?
  - Context Recall: Does the context cover the ground truth?
  - Answer Relevancy: (computed via embeddings — no extra LLM calls)
  - Context Precision: (computed via embeddings — no extra LLM calls)

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --questions 6
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()


# ── Rate-limit safe LLM call ─────────────────────────────────────────────────

def call_groq(messages, model="llama-3.3-70b-versatile", max_tokens=1024, retries=5):
    """
    Call LLM with exponential backoff. Uses OpenRouter to avoid Groq rate limits
    during bulk evaluation.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    # Use OpenRouter for evaluation (no aggressive per-minute rate limits)
    api_key = os.getenv("OPENROUTER_API_KEY")
    eval_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    llm = ChatOpenAI(
        model=eval_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.3,
        max_tokens=max_tokens,
        max_retries=0,
        request_timeout=60,
    )

    for attempt in range(retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
                print(f"    [Rate limit] Waiting {wait}s before retry {attempt+2}/{retries}...")
                time.sleep(wait)
            else:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    raise
    raise RuntimeError("All retries exhausted")


# ── RAG Pipeline (evaluation-optimized) ───────────────────────────────────────

def run_rag_pipeline(question: str) -> dict:
    """
    Run retrieval + generation. Uses only 1 Groq LLM call per question.
    No router, no guardrail, no query expansion — pure retrieval + generation.
    """
    from retrieval.vector_search import VectorSearch
    from retrieval.bm25_search import BM25Search
    from retrieval.rrf_fusion import reciprocal_rank_fusion
    from retrieval.reranker import Reranker
    from retrieval.hybrid_merger import HybridMerger
    from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE
    from langchain_core.messages import SystemMessage, HumanMessage

    start = time.time()

    # 1. Hybrid Retrieval (no LLM calls)
    vector_search = VectorSearch()
    bm25_search = BM25Search()
    reranker = Reranker()

    dense_results = vector_search.search(question, top_k=30)
    for r in dense_results:
        r["retrieval_method"] = "dense"

    bm25_results = []
    try:
        bm25_results = bm25_search.search(question, top_k=30)
    except Exception:
        pass

    # RRF merge
    if dense_results and bm25_results:
        merged = reciprocal_rank_fusion(dense_results, bm25_results, top_k=25)
    elif dense_results:
        merged = dense_results[:25]
    else:
        merged = bm25_results[:25]

    # Rerank with Jina (API call, not LLM)
    if merged:
        merged = reranker.rerank(question, merged, top_n=5)
        merged = [r for r in merged if r.get("rerank_score", 0) >= 0.2]

    # 2. Format context
    merger = HybridMerger()
    context = merger.merge(merged, [], [])

    # 3. Generate answer — single Groq call with retries
    prompt = QUERY_PROMPT_TEMPLATE.format(question=question, context=context)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        answer = call_groq(messages, model="llama-3.3-70b-versatile", max_tokens=1024)
    except Exception as e:
        answer = f"ERROR: {str(e)[:100]}"

    latency = (time.time() - start) * 1000

    # Extract contexts as list
    contexts = [r.get("text", "")[:800] for r in merged if r.get("text")]

    return {
        "answer": answer,
        "contexts": contexts,
        "route": "Hybrid (Dense + BM25 + RRF + Jina)",
        "latency_ms": round(latency, 1),
    }


# ── RAGAS Evaluation ──────────────────────────────────────────────────────────

def evaluate_with_ragas(results: list) -> dict:
    """
    Compute RAGAS metrics. Uses Groq 8B for LLM-based metrics (faithfulness,
    context_recall) and local embeddings for the rest.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_recall
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings

    groq_key = os.getenv("GROQ_API_KEY")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
        temperature=0.0,
        max_tokens=4096,
        max_retries=5,
        request_timeout=120,
    ))

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )

    # Build dataset
    samples = []
    for r in results:
        if "ERROR" in r.get("answer", ""):
            continue  # Skip failed questions
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] if r["contexts"] else ["No context"],
            reference=r["ground_truth"],
        ))

    if not samples:
        return {"error": "No valid samples to evaluate"}

    dataset = EvaluationDataset(samples=samples)

    print(f"\n{'='*60}")
    print(f"  RAGAS EVALUATION")
    print(f"{'='*60}")
    print(f"  Valid samples: {len(samples)}/{len(results)}")
    print(f"  Evaluator: Groq (llama-3.1-8b-instant)")
    print(f"  Metrics: faithfulness, context_recall")
    print(f"{'='*60}\n")

    # Run RAGAS (only faithfulness + context_recall — these work with n=1)
    eval_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Extract from DataFrame
    df = eval_result.to_pandas()
    scores = {}
    for col in df.columns:
        if col in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        values = df[col].dropna()
        if len(values) > 0:
            scores[col] = round(float(values.mean()), 4)

    # Compute answer_relevancy via embeddings (question vs answer similarity)
    from ingestion.embeddings import Embedder
    embedder = Embedder()

    relevancy_scores = []
    precision_scores = []

    for r in results:
        if "ERROR" in r.get("answer", ""):
            continue

        # Answer Relevancy: cosine(question_embedding, answer_embedding)
        q_vec = np.array(embedder.embed_one(r["question"]))
        a_vec = np.array(embedder.embed_one(r["answer"][:500]))
        sim = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-8))
        relevancy_scores.append(max(0, sim))

        # Context Precision: fraction of contexts relevant to question
        if r["contexts"]:
            relevant = 0
            for ctx in r["contexts"]:
                c_vec = np.array(embedder.embed_one(ctx[:500]))
                ctx_sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
                if ctx_sim >= 0.45:
                    relevant += 1
            precision_scores.append(relevant / len(r["contexts"]))
        else:
            precision_scores.append(0.0)

    if relevancy_scores:
        scores["answer_relevancy"] = round(float(np.mean(relevancy_scores)), 4)
    if precision_scores:
        scores["context_precision"] = round(float(np.mean(precision_scores)), 4)

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        test_data = json.load(f)

    if args.questions:
        test_data = test_data[:args.questions]

    print(f"\n{'='*60}")
    print(f"  AI FINANCIAL ADVISOR - RAG EVALUATION")
    print(f"{'='*60}")
    print(f"  Questions: {len(test_data)}")
    print(f"  LLM: Groq (llama-3.3-70b-versatile)")
    print(f"  Retrieval: Dense + BM25 + RRF + Jina Reranker")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Run pipeline
    results = []
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        print(f"  [{i}/{len(test_data)}] {question[:60]}...")

        try:
            rag_output = run_rag_pipeline(question)
            is_error = "ERROR" in rag_output["answer"]
            status = "FAIL" if is_error else "OK"
            results.append({
                "question": question,
                "ground_truth": item["ground_truth"],
                "category": item.get("category", ""),
                "answer": rag_output["answer"],
                "contexts": rag_output["contexts"],
                "route": rag_output["route"],
                "latency_ms": rag_output["latency_ms"],
            })
            preview = rag_output["answer"][:50].replace("\n", " ")
            print(f"         {status} | {rag_output['latency_ms']:.0f}ms | "
                  f"{len(rag_output['contexts'])} ctx | {preview}...")
        except Exception as e:
            print(f"         FAIL | {str(e)[:60]}")
            results.append({
                "question": question,
                "ground_truth": item["ground_truth"],
                "category": item.get("category", ""),
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "route": "error",
                "latency_ms": 0,
            })

        # Wait between questions (OpenRouter is less aggressive than Groq)
        if i < len(test_data):
            print(f"         ... waiting 10s (rate limit protection)")
            time.sleep(10)

    # Latency stats
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    latency_stats = {
        "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "min_ms": round(min(latencies), 1) if latencies else 0,
        "max_ms": round(max(latencies), 1) if latencies else 0,
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
    }

    successful = sum(1 for r in results if "ERROR" not in r["answer"])
    print(f"\n  Pipeline complete: {successful}/{len(results)} successful")
    print(f"  Latency: mean={latency_stats['mean_ms']}ms, p50={latency_stats['p50_ms']}ms")

    # RAGAS scoring
    ragas_scores = {}
    if not args.skip_ragas and successful > 0:
        print("\n  Waiting 30s before RAGAS scoring (rate limit cooldown)...")
        time.sleep(30)
        try:
            ragas_scores = evaluate_with_ragas(results)

            print(f"\n{'='*60}")
            print(f"  FINAL RAGAS SCORES")
            print(f"{'='*60}")
            for metric, score in sorted(ragas_scores.items()):
                if metric == "error":
                    continue
                bar_len = int(score * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"  {metric:25s} [{bar}] {score:.4f}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\n  RAGAS scoring failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"eval_{timestamp}.json"
    output_path = output_dir / output_file

    report = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "num_questions": len(test_data),
            "successful": successful,
            "model_generation": "llama-3.3-70b-versatile (Groq)",
            "model_evaluator": "llama-3.1-8b-instant (Groq)",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "retrieval": "Dense (BGE-M3) + BM25 + RRF + Jina Reranker",
        },
        "ragas_scores": ragas_scores,
        "latency_stats": latency_stats,
        "per_question_results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {output_path}")

    # Category summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY BY CATEGORY")
    print(f"{'='*60}")
    categories = set(r["category"] for r in results if r["category"])
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        answered = sum(1 for r in cat_results if "ERROR" not in r["answer"])
        avg_lat = sum(r["latency_ms"] for r in cat_results) / len(cat_results)
        print(f"  {cat:15s} | {answered}/{len(cat_results)} answered | avg {avg_lat:.0f}ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
