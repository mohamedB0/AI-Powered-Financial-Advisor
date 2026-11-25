"""
agents/financial_agent.py
LangGraph-based financial analysis agent.

Uses:
  - OpenRouter API for LLM inference (qwen/qwen-2.5-72b-instruct)
  - Query Router for routing to SQL/Web/Vector
  - LangGraph StateGraph for agentic orchestration
"""

import sys
import time
import concurrent.futures
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from retrieval.query_router import route_query, QueryRoute
from retrieval.vector_search import VectorSearch
from retrieval.sql_search import SQLSearch
from retrieval.web_search import WebSearch
from retrieval.hybrid_merger import HybridMerger
from retrieval.reranker import Reranker
from retrieval.bm25_search import BM25Search
from retrieval.rrf_fusion import reciprocal_rank_fusion
from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE, GRADER_PROMPT, REWRITE_PROMPT, GUARDRAIL_PROMPT, CONTEXTUALIZE_PROMPT


# â”€â”€ Agent State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AgentState(TypedDict):
    messages         : Annotated[List[BaseMessage], lambda a, b: a + b]
    history          : List[BaseMessage]          # prior conversation turns
    question         : str
    original_question: str                        # Anchor for language matching
    context          : str
    answer           : str
    route            : str
    context_relevant : str
    is_safe          : str                        # CRAG guard: safety flag
    retry_count      : int                        # CRAG guard: max 2 rewrites
    top_n            : int
    persona          : str
    focus_mode       : str
    collection       : str                        # Target collection: "all", "financial_data", "user_data"


# â”€â”€ LLM Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€ Speed: Groq is ~10-20x faster than OpenRouter for equivalent models â”€â”€â”€â”€â”€â”€â”€
# Primary path: Groq (ultra-fast inference)
# Fallback path: OpenRouter (if Groq key missing or quota exceeded)

_GROQ_MODELS = [
    "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile",   # Fastest high-quality Groq model
    "llama-3.1-8b-instant",       # Ultra-fast fallback
]

_FALLBACK_MODELS = [
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct:free",
]

def _get_llm(model: str | None = None):
    """Returns the best available LLM. Prefers Groq for speed."""
    # âš¡ Read at call-time so a fresh GROQ_API_KEY in .env takes effect
    # without needing to restart the worker.
    groq_key = os.getenv("GROQ_API_KEY")
    # Anti-repetition penalties: Llama 3.3 70B has a known habit of looping
    # on Arabic / minority languages. Both penalties damp that out without
    # hurting French or English quality.
    _anti_loop = {"frequency_penalty": 0.4, "presence_penalty": 0.2}
    if groq_key and model is None:
        # Use Groq primary model â€” significantly faster inference
        groq_model = os.getenv("GROQ_MODEL", _GROQ_MODELS[0])
        return ChatOpenAI(
            model=groq_model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.3, max_tokens=2048, max_retries=0, request_timeout=20,
            model_kwargs=_anti_loop,
        )
    # If the requested model is one of the known Groq models, use Groq.
    if model in _GROQ_MODELS and groq_key:
        return ChatOpenAI(
            model=model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.3, max_tokens=2048, max_retries=0, request_timeout=20,
            model_kwargs=_anti_loop,
        )
    # Try Groq fallback key on the same primary model â€” handles rate limits
    # without paying the 60s+ OpenRouter latency.
    if model == "_GROQ_FALLBACK_KEY":
        fallback_key = os.getenv("GROQ_API_KEY_Fallback")
        if fallback_key:
            return ChatOpenAI(
                model=os.getenv("GROQ_MODEL", _GROQ_MODELS[0]),
                base_url="https://api.groq.com/openai/v1",
                api_key=fallback_key,
                temperature=0.3, max_tokens=2048, max_retries=0, request_timeout=20,
                model_kwargs=_anti_loop,
            )
    # OpenRouter fallback (or explicit model override)
    if model is None:
        model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3, max_tokens=2048, max_retries=0, request_timeout=60,
        model_kwargs=_anti_loop,
    )

def _get_fast_llm():
    """Returns a fast/cheap LLM for guardrails, routing, rewriting etc. Prefers Groq."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.0, max_tokens=256, max_retries=0, request_timeout=8,
        )
    model = os.getenv("FAST_LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
    return ChatOpenAI(
        model=model, base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0, max_tokens=256, max_retries=0, request_timeout=15,
    )


# â”€â”€ Lazy-loaded retrieval singletons (avoid 25s+ init at import time) â”€â”€â”€â”€â”€â”€â”€â”€â”€

import threading
_init_lock = threading.Lock()

_vector_search = None
_sql_search = None
_web_search = None
_merger = None

def _get_vector_search():
    global _vector_search
    if _vector_search is None:
        with _init_lock:
            if _vector_search is None:
                _vector_search = VectorSearch()
    return _vector_search

def _get_sql_search():
    global _sql_search
    if _sql_search is None:
        with _init_lock:
            if _sql_search is None:
                _sql_search = SQLSearch()
    return _sql_search

def _get_web_search():
    global _web_search
    if _web_search is None:
        with _init_lock:
            if _web_search is None:
                _web_search = WebSearch()
    return _web_search

def _get_merger():
    global _merger
    if _merger is None:
        with _init_lock:
            if _merger is None:
                _merger = HybridMerger()
    return _merger

_reranker = None
def _get_reranker():
    global _reranker
    if _reranker is None:
        with _init_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker

_bm25_search = None
def _get_bm25_search():
    global _bm25_search
    if _bm25_search is None:
        with _init_lock:
            if _bm25_search is None:
                _bm25_search = BM25Search()
    return _bm25_search


UNSUPPORTED_REGIONS: list = []  # All regions supported; BVC tickers are in the DB
US_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]


def _results_match_query(results: list, question: str) -> bool:
    """Check if SQL results actually match the query (ticker AND data type).

    Returns False for vague/conceptual questions that have no concrete
    structured-data intent, so generic market dumps never leak into answers.
    """
    if not results:
        return False
    q = question.lower()
    result_str = str(results).upper()

    is_revenue_query = any(k in q for k in ["revenue", "earning", "income", "ebitda", "eps"])
    is_price_query = any(k in q for k in ["price", "close", "open", "high", "low", "volume", "stock"])

    if is_revenue_query:
        has_financial_data = any(word in result_str for word in ["METRIC_NAME", "VALUE", "PERIOD", "REVENUE", "EBITDA", "EPS"])
        if not has_financial_data:
            print(f"[Router] Query asks for financial metrics but got stock prices")
            return False
    elif is_price_query:
        has_price_data = any(word in result_str for word in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DATE"])
        if not has_price_data:
            print(f"[Router] Query asks for price data but got other data")
            return False

    # Ticker-specific questions: only keep results that actually mention the ticker.
    if "atw" in q or "attijari" in q:
        return any("ATW" in str(r) or "BVC.ATW" in str(r) for r in results)
    if "bcm" in q or "bcp" in q or "banque populaire" in q:
        return any("BCP" in str(r) or "BVC.BCP" in str(r) for r in results)
    if "iam" in q or "maroc telecom" in q:
        return any("IAM" in str(r) or "BVC.IAM" in str(r) for r in results)
    for ticker in US_TICKERS:
        if ticker.lower() in q:
            return any(ticker in str(r) for r in results)

    # No ticker mentioned. Only accept generic market data when the question
    # expresses a concrete market-data intent (movers, prices, revenue, volume).
    # Conceptual/definitional questions ("how to measure inflation") fall here
    # and are rejected so irrelevant tables don't pollute the context.
    # NOTE: deliberately excludes broad words like "market"/"marchÃ©", since a
    # conceptual question can mention "the market" without wanting price data.
    generic_market_intent = (
        is_revenue_query
        or is_price_query
        or any(k in q for k in [
            "mover", "gainer", "loser", "perform", "trending",
            "most active", "rising", "falling",
        ])
    )
    if not generic_market_intent:
        print("[Router] SQL results have no concrete relevance to the question â€” discarding.")
        return False
    return True


def retrieve_context(state: AgentState) -> AgentState:
    """
    Node: Retrieve context using ROUTE-AWARE source selection.

    The router decides which sources are relevant. We only query the sources
    that match the route to avoid bloating the context with irrelevant data
    (e.g., AAPL stock prices for a Morocco market query).

    Routes:
      - SQL:    SQL only (+ vector as supplement)
      - WEB:    Web only
      - VECTOR: Vector only
      - HYBRID: SQL + Vector + Web
    """
    question = state["question"]

    routing = route_query(question)
    route = routing["route"]
    q_lower = question.lower()

    vector_results = []
    sql_results = []
    web_results = []

    # Dynamically respect the Semantic Router's decisions
    required_tools = routing.get("required_tools", [])
    need_sql    = "sql" in required_tools
    need_vector = "vector" in required_tools
    need_web    = "web_search" in required_tools
    
    if "hybrid" in str(route).lower():
        need_vector = True
        need_web = True

    # APPLY FOCUS MODE OVERRIDES
    focus_mode = state.get("focus_mode", "New Chat").lower()
    if "company" in focus_mode:
        need_sql = True
        need_vector = False
        need_web = False
        print("[Router] FOCUS MODE: Company Analysis. Forcing SQL Search.")
    elif "macro" in focus_mode:
        need_sql = False
        need_vector = True
        need_web = False
        print("[Router] FOCUS MODE: Market Macro. Forcing Vector Search.")

    # â”€â”€ 1. SQL Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if need_sql:
        try:
            # Explicit intent detection â€” we only hit the structured DB when the
            # question actually asks for market/financial figures. A vague or
            # conceptual question (e.g. "how to measure inflation") must NOT
            # trigger a generic top-movers dump that then leaks into the answer.
            movers_intent = any(k in q_lower for k in [
                "mover", "gainer", "loser", "top perform", "best perform",
                "worst perform", "biggest gain", "biggest los", "rising", "falling",
                "trending", "most active", "top stock", "best stock",
            ])

            # Comparison detection: "compare X vs Y", "X contre Y", "Ù…Ù‚Ø§Ø±Ù†Ø©"
            compare_intent = any(k in q_lower for k in [
                "compare", "compar", "versus", " vs ", " vs.", "contre",
                "Ù…Ù‚Ø§Ø±Ù†Ø©", "Ù‚Ø§Ø±Ù†", "difference between", "diffÃ©rence entre",
            ])

            if compare_intent and routing.get("detail"):
                # Try to extract two tickers from the question
                import re
                tickers_found = re.findall(r'\b([A-Z]{2,5})\b', question)
                if len(tickers_found) >= 2:
                    print(f"[Router] Comparison detected: {tickers_found[0]} vs {tickers_found[1]}")
                    comparison_text = _get_sql_search().compare_companies(tickers_found[0], tickers_found[1])
                    sql_results = [{"comparison": comparison_text}]
                else:
                    sql_results = _get_sql_search().get_price_history(routing["detail"], days=10)
            elif routing["detail"]:
                sql_results = _get_sql_search().get_price_history(routing["detail"], days=10)
            elif "revenue" in q_lower or "earnings" in q_lower:
                sql_results = _get_sql_search().get_top_by_revenue()
            elif "volume" in q_lower:
                sql_results = _get_sql_search().get_highest_volume()
            elif movers_intent:
                sql_results = _get_sql_search().get_top_movers()
            elif compare_intent:
                # Comparison without a specific detail â€” try to find tickers in the question
                import re
                tickers_found = re.findall(r'\b([A-Z]{2,5})\b', question)
                if len(tickers_found) >= 2:
                    print(f"[Router] Comparison detected: {tickers_found[0]} vs {tickers_found[1]}")
                    comparison_text = _get_sql_search().compare_companies(tickers_found[0], tickers_found[1])
                    sql_results = [{"comparison": comparison_text}]
                else:
                    sql_results = []
            else:
                # No concrete structured-data intent â€” skip SQL entirely rather
                # than fabricating an irrelevant table.
                print("[Router] SQL routed but no concrete data intent â€” skipping SQL fetch.")
                sql_results = []

            # Validate: do SQL results actually match what the user asked?
            if sql_results and not _results_match_query(sql_results, question):
                print(f"[Router] SQL results don't match query â€” discarding")
                sql_results = []
        except Exception as e:
            print(f"[Router] SQL error: {e}")

    # â”€â”€ 2. Hybrid Retrieval: Dense (BGE-M3) + Sparse (BM25) + RRF + Reranker â”€â”€
    vector_quality = "none"  # Track: "none", "weak", "strong"
    # Detect if user explicitly selected a specific collection (not "all")
    user_forced_collection = state.get("collection", "all") != "all"
    if need_vector:
        # Determine target collection(s) based on user selection
        target_collection = state.get("collection", "all")
        coll_filter = None if target_collection == "all" else target_collection

        try:
            # 2a. Dense retrieval (semantic embeddings via ChromaDB) â€” wider pool for better recall
            # Use more candidates for user-selected collections (smaller, need higher recall)
            # âš¡ Speed: trimmed pool for institutional collection (BM25 already covers keyword variants)
            search_top_k = 40 if user_forced_collection else 20
            dense_results = _get_vector_search().search(question, top_k=search_top_k, collection=coll_filter)
            # Tag results for RRF source tracking
            for r in dense_results:
                r["retrieval_method"] = "dense"

            # 2b. Sparse retrieval (BM25 keyword matching) â€” wider pool
            bm25_results = []
            try:
                bm25_results = _get_bm25_search().search(question, top_k=15)
            except Exception as e:
                print(f"[Router] BM25 search error (non-fatal): {e}")

            # 2c. Reciprocal Rank Fusion â€” merge all retrieval streams
            all_streams = [s for s in [dense_results, bm25_results] if s]
            if all_streams:
                if len(all_streams) > 1:
                    total_docs = sum(len(s) for s in all_streams)
                    print(f"[Router] Hybrid retrieval: {total_docs} total docs from {len(all_streams)} streams -> RRF merge")
                    # âš¡ Trimmed fused pool for faster reranker call
                    vector_results = reciprocal_rank_fusion(*all_streams, top_k=15)
                else:
                    vector_results = all_streams[0][:15]

            # 2e. Rerank fused results with Jina cross-encoder
            #     âš¡ Skip when one dense result clearly dominates (saves API roundtrip).
            if vector_results:
                dynamic_top_n = state.get("top_n", 5)

                # Fast-path: if dense top score is already very high and gap to #2 is large,
                # the reranker would just confirm the order â€” skip it.
                _can_skip_rerank = False
                if dense_results and len(dense_results) >= 2:
                    s1 = dense_results[0].get("score", 0)
                    s2 = dense_results[1].get("score", 0)
                    if s1 >= 0.85 and (s1 - s2) >= 0.15 and not user_forced_collection:
                        _can_skip_rerank = True
                        print(f"[Router] Skipping reranker â€” dense top dominates ({s1:.2f} vs {s2:.2f})")
                        vector_results = vector_results[:dynamic_top_n]
                        # Synthesise a rerank_score so downstream filters/quality keep working
                        for r in vector_results:
                            r["rerank_score"] = r.get("score", 0.5)

                if not _can_skip_rerank:
                    print(f"[Router] Reranking {len(vector_results)} hybrid results via Jina API (top_n={dynamic_top_n})...")
                    vector_results = _get_reranker().rerank(question, vector_results, top_n=dynamic_top_n)

                # 2f. Filter out low-confidence results (improves Context Precision)
                # Use a lower threshold for user-selected collections â€” the user
                # knows their document has the answer, so be more permissive.
                if vector_results:
                    min_score = 0.08 if user_forced_collection else 0.25
                    before_filter = len(vector_results)
                    vector_results = [r for r in vector_results if r.get("rerank_score", 0) >= min_score]
                    if len(vector_results) < before_filter:
                        print(f"[Router] Filtered {before_filter - len(vector_results)} low-confidence results (< {min_score})")

                # Check reranker confidence
                if vector_results:
                    best_score = max(r.get("rerank_score", 0) for r in vector_results)
                    print(f"[Router] Best reranker score: {best_score:.4f} ({len(vector_results)} docs)")
                    # For user collections, trust any result (the user chose this collection)
                    if user_forced_collection:
                        vector_quality = "strong"
                    else:
                        vector_quality = "strong" if best_score >= 0.3 else "weak"

            # 2g. Context window expansion for user collections â€” fetch neighboring
            # chunks (Â±1) to give the LLM surrounding context for fragmented chunks.
            if user_forced_collection and vector_results:
                try:
                    expanded = []
                    seen_ids = set()
                    for r in vector_results:
                        source = r.get("source_file", "")
                        idx = r.get("chunk_index")
                        expanded.append(r)
                        seen_ids.add(f"{source}_{idx}")

                        # Fetch neighbors from same document
                        if idx is not None and source:
                            for neighbor_idx in [idx - 1, idx + 1]:
                                nid = f"{source}_{neighbor_idx}"
                                if nid in seen_ids or neighbor_idx < 0:
                                    continue
                                seen_ids.add(nid)
                                neighbors = _get_vector_search().search(
                                    "", top_k=1, collection=coll_filter,
                                    filter_metadata={"source_file": source, "chunk_index": neighbor_idx}
                                )
                                if neighbors:
                                    neighbors[0]["score"] = r.get("score", 0) * 0.8  # slightly lower
                                    expanded.append(neighbors[0])
                    if len(expanded) > len(vector_results):
                        print(f"[Router] Context expansion: {len(vector_results)} â†’ {len(expanded)} chunks")
                        vector_results = expanded
                except Exception as e:
                    print(f"[Router] Context expansion error (non-fatal): {e}")

        except Exception as e:
            print(f"[Router] Hybrid retrieval error: {e}")

    # â”€â”€ 3. Web Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #   Always runs if:
    #     a) Router explicitly asked for web  OR
    #     b) Vector results are weak/absent (auto-supplement to avoid "je n'ai pas trouvÃ©")
    #   BUT: if the user explicitly selected a specific collection (not "all"),
    #   respect their choice and do NOT override with web results.

    if need_web and not user_forced_collection:
        try:
            web_results = _get_web_search().search(question, max_results=3)
        except Exception as e:
            print(f"[Router] Web search error: {e}")
    elif not user_forced_collection and vector_quality in ("none", "weak"):
        print(f"[Router] Vector quality '{vector_quality}' â€” auto-supplementing with web search.")
        try:
            web_results = _get_web_search().search(question, max_results=3)
        except Exception as e:
            print(f"[Router] Web search error: {e}")
    elif user_forced_collection:
        print(f"[Router] User selected specific collection '{state.get('collection')}' â€” skipping web supplement.")

    # â”€â”€ 4. Merge only the sources that were fetched â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # If vector quality is weak but web returned results, drop weak vectors to avoid confusion
    # BUT: if user explicitly chose a collection, always keep their vector results.
    if vector_quality == "weak" and web_results and not user_forced_collection:
        print(f"[Router] Dropping {len(vector_results)} weak vector results in favor of web.")
        vector_results = []

    context = _get_merger().merge(vector_results, sql_results, web_results)

    # Determine dynamic route name for the UI label
    executed = []
    if sql_results: executed.append("SQL")
    if vector_results: executed.append("Vector")
    if web_results: executed.append("Web Search")
    route_label = " + ".join(executed) if executed else "No Sources"

    return {
        "context": context,
        "route": route_label,
    }


def _detect_language(text: str) -> str:
    """Simple language detection for French/Arabic/English."""
    # Arabic script detection
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "Arabic"
    # French indicators
    french_words = {"le", "la", "les", "des", "est", "que", "qui", "dans", "pour", "sur", "pas", "une", "avec", "sont", "cette", "quel", "quelle", "comment", "pourquoi"}
    words = set(text.lower().split())
    if len(words & french_words) >= 2:
        return "French"
    return "English"


def generate_answer(state: AgentState) -> AgentState:
    """Node: Generate answer using LLM with retrieved context."""
    # CRITICAL: Always use the 'original_question' for generation.
    # The 'question' might have been rewritten/translated by the LLM in english.
    # Using original_question guarantees the LLM sees the user's explicit Arabic/French text.
    original_q = state["original_question"]
    detected_lang = _detect_language(original_q)

    prompt = QUERY_PROMPT_TEMPLATE.format(
        question=original_q,
        context=state["context"] or "No context available.",
    )

    # Add explicit language enforcement at the end of the prompt
    lang_instruction = f"\n\nâš ï¸ MANDATORY: Your entire response MUST be written in {detected_lang}. The user asked in {detected_lang}, so reply ONLY in {detected_lang}."
    prompt += lang_instruction

    # Prepend prior conversation history (last 4 messages = 2 turns) to save tokens
    prior = state.get("history", [])[-4:]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *prior,
        HumanMessage(content=prompt),
    ]

    # âš¡ Speed: Try Groq first (None â†’ uses Groq if key present), then Groq
    # fallback key (handles rate limits without 60s+ OpenRouter latency),
    # then OpenRouter fallbacks. Passing None lets _get_llm() pick Groq automatically.
    openrouter_primary = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
    # (None sentinel = Groq preferred path)
    models_to_try = (
        [None]
        + (["_GROQ_FALLBACK_KEY"] if os.getenv("GROQ_API_KEY_Fallback") else [])
        + [m for m in _FALLBACK_MODELS if m != openrouter_primary]
        + [openrouter_primary]
    )

    for model_name in models_to_try:
        current_llm = _get_llm(model_name)
        # Retry with exponential backoff on rate limit (up to 4 attempts for Groq)
        for attempt in range(4):
            try:
                response = current_llm.invoke(messages)
                if model_name is not None:  # None = Groq (primary), log only for fallbacks
                    print(f"[LLM] Successfully used fallback model: {model_name}")
                return {
                    "messages": [response],
                    "answer"  : response.content,
                }
            except Exception as e:
                error_msg = str(e)

                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    if attempt < 3:
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                        print(f"[LLM] Rate limit on {model_name}. Retrying in {wait}s... ({attempt+1}/4)")
                        time.sleep(wait)
                        continue
                    print(f"[LLM] Rate limit exhausted for {model_name}.")
                    break

                if "provider" in error_msg.lower() or "502" in error_msg or "503" in error_msg:
                    print(f"[LLM] Provider error on {model_name}: {error_msg[:120]}. Trying next model.")
                    break  # try next model

                if "tool_use_failed" in error_msg or "failed_generation" in error_msg:
                    return {
                        "messages": [],
                        "answer": "I encountered an issue processing your request. Please try rephrasing your question.",
                    }

                # Other unexpected error â€” log and try next model
                print(f"[LLM] Unexpected error on {model_name} attempt {attempt + 1}: {error_msg[:120]}")
                if attempt == 0:
                    time.sleep(1)
                    continue
                break  # try next model

    # All models exhausted
    fallback = (
        "**âš ï¸ Service Temporarily Unavailable**\n\n"
        "The AI provider is currently experiencing issues. "
        "Please try again in a moment.\n\n"
        "Here is the raw evidence retrieved from your documents:\n\n"
        f"{state['context'] or 'No context available.'}"
    )
    return {"messages": [], "answer": fallback}


# â”€â”€ Guardrail Node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_OBVIOUSLY_SAFE_PATTERNS = [
    # Finance / econ terms in French, English, Arabic
    "marchÃ©", "bourse", "action", "obligation", "inflation", "taux", "pib", "croissance",
    "dividende", "rendement", "risque", "portefeuille", "investissement", "bÃ©nÃ©fice",
    "stock", "market", "price", "revenue", "gdp", "interest", "bond", "equity",
    "earnings", "income", "ebitda", "eps", "share", "dividend", "profit", "margin",
    "debt", "asset", "liability", "balance sheet", "cash flow", "annual report",
    "quarter", "fiscal", "segment", "operating", "net sales", "gross margin",
    "company", "financial", "capital", "valuation", "p/e", "ratio",
    "mosaic", "ocp", "attijariwafa", "bcp", "iam", "managem", "cosumar",
    # Arabic finance / market terms (covers ~95% of typical Arabic queries)
    "Ø³Ù‡Ù…", "Ø£Ø³Ù‡Ù…", "Ø³ÙˆÙ‚", "Ø£Ø³ÙˆØ§Ù‚", "ØªØ¶Ø®Ù…", "ÙØ§Ø¦Ø¯Ø©", "Ù†Ø§ØªØ¬", "Ø§Ø³ØªØ«Ù…Ø§Ø±", "Ø¹Ø§Ø¦Ø¯",
    "Ø£Ø±Ø¨Ø§Ø­", "Ø¥ÙŠØ±Ø§Ø¯Ø§Øª", "Ø´Ø±ÙƒØ©", "Ø´Ø±ÙƒØ§Øª", "Ø¨ÙˆØ±ØµØ©", "Ø¥Ø¯Ø±Ø§Ø¬", "Ø§Ù‚ØªØµØ§Ø¯", "Ø§Ù‚ØªØµØ§Ø¯ÙŠ",
    "Ø¨Ù†Ùƒ", "Ø§Ù„Ø¨Ù†Ùƒ", "Ù…ØµØ±Ù", "Ù…ØµØ±ÙÙŠ", "Ù‚Ø±Ø¶", "Ù‚Ø±ÙˆØ¶", "ØµÙ†Ø¯ÙˆÙ‚", "Ù…ÙŠØ²Ø§Ù†ÙŠØ©", "Ù…Ø§Ù„ÙŠ",
    "Ù…Ø§Ù„ÙŠØ©", "ØªÙ…ÙˆÙŠÙ„", "Ø£ØµÙˆÙ„", "Ø®ØµÙˆÙ…", "Ø±Ø¨Ø­ÙŠØ©", "Ù†Ù…Ùˆ", "Ø±ÙƒÙˆØ¯", "ØªØ¯Ø§ÙˆÙ„", "Ø³Ù†Ø¯",
    "Ø³Ù†Ø¯Ø§Øª", "Ø£Ø¯Ø§Ø¡", "Ù…Ø¤Ø´Ø±", "Ø§Ù„Ø¯Ø§Ø± Ø§Ù„Ø¨ÙŠØ¶Ø§Ø¡", "Ø§Ù„Ù…ØºØ±Ø¨", "Ø¨Ù†Ùƒ Ø§Ù„Ù…ØºØ±Ø¨",
    "budget", "dette", "fiscal", "hcp", "bank al-maghrib", "ammc", "bkam",
]

# âš ï¸ Red-flag patterns: even if a question contains safe financial keywords,
# these intent markers force the full LLM safety check (or direct refusal).
# Covers tax evasion, money laundering, insider trading, scam attempts,
# and other malicious intent in FR/EN/AR.
_UNSAFE_INTENT_PATTERNS = [
    # English â€” illegal intent
    "how to evade", "how to avoid tax", "tax evasion", "money laundering",
    "launder money", "insider trading", "manipulate the market", "pump and dump",
    "ponzi", "scam", "fraud", "embezzle", "bribe", "kickback",
    "hide money", "hide income", "offshore evasion", "shell company to hide",
    # French â€” intention illÃ©gale
    "comment Ã©chapper", "Ã©chapper aux impÃ´ts", "Ã©chapper Ã  l'impÃ´t",
    "Ã©vasion fiscale", "fraude fiscale", "blanchiment", "blanchir",
    "dÃ©lit d'initiÃ©", "manipuler le marchÃ©", "cacher de l'argent",
    "cacher des revenus", "dissimuler", "frauder le fisc",
    "comment frauder", "dÃ©tournement de fonds", "pot-de-vin", "corruption",
    # Arabic â€” Ù†ÙŠØ© ØºÙŠØ± Ù…Ø´Ø±ÙˆØ¹Ø©
    "ÙƒÙŠÙ Ø£Ù‡Ø±Ø¨ Ù…Ù† Ø§Ù„Ø¶Ø±Ø§Ø¦Ø¨", "Ø§Ù„Ù‡Ø±ÙˆØ¨ Ù…Ù† Ø§Ù„Ø¶Ø±Ø§Ø¦Ø¨", "Ø§Ù„ØªÙ‡Ø±Ø¨ Ø§Ù„Ø¶Ø±ÙŠØ¨ÙŠ",
    "ØºØ³ÙŠÙ„ Ø§Ù„Ø£Ù…ÙˆØ§Ù„", "ØºØ³Ù„ Ø§Ù„Ø£Ù…ÙˆØ§Ù„", "Ø§Ù„Ø§Ø­ØªÙŠØ§Ù„", "Ø§Ø­ØªÙŠØ§Ù„", "Ù†ØµØ¨",
    "Ø¥Ø®ÙØ§Ø¡ Ø§Ù„Ø£Ù…ÙˆØ§Ù„", "Ø¥Ø®ÙØ§Ø¡ Ø§Ù„Ø¯Ø®Ù„", "ÙƒÙŠÙ Ø£Ø®ÙÙŠ", "ÙƒÙŠÙ Ø£Ø­ØªØ§Ù„",
    "ØªØ¨ÙŠÙŠØ¶ Ø§Ù„Ø§Ù…ÙˆØ§Ù„","Ø§Ù„ØªØ¯Ø§ÙˆÙ„ Ù…Ù† Ø§Ù„Ø¯Ø§Ø®Ù„", "Ø§Ù„ØªÙ„Ø§Ø¹Ø¨ Ø¨Ø§Ù„Ø³ÙˆÙ‚", "Ø§Ø®ØªÙ„Ø§Ø³", "Ø±Ø´ÙˆØ©",
]

# Greetings and casual conversational phrases -- always safe.
# These are handled with a friendly response rather than the full retrieval pipeline.
_GREETING_PATTERNS = [
    # English
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "howdy", "greetings", "thanks", "thank you",
    "who are you", "what can you do", "help me", "what are you",
    # French
    "bonjour", "bonsoir", "salut", "coucou", "merci", "comment ca va",
    "bonne journee", "bienvenue", "qui es-tu", "que fais-tu",
    # Darija / Arabic
    "salam", "slm", "labas", "cv", "ahlan", "marhaba", "choukran",
]


def _is_greeting(question_lower: str) -> bool:
    """Check if the query is a simple greeting/conversational phrase."""
    q = question_lower.strip().rstrip("?!.,")
    # Exact match or very short greeting
    if q in _GREETING_PATTERNS:
        return True
    # Starts with a greeting and is short (< 30 chars)
    if len(q) < 30 and any(q.startswith(g) for g in _GREETING_PATTERNS):
        return True
    return False


def _generate_greeting_response(question: str) -> str:
    """Generate a friendly greeting response."""
    from agents.financial_agent import _detect_language
    lang = _detect_language(question)
    if lang == "French":
        return (
            "Bonjour ! Je suis votre assistant d'analyse financiere. "
            "Je peux vous aider avec :\n\n"
            "- **Analyse des marches** : cours en direct de la Bourse de Casablanca\n"
            "- **Donnees financieres** : revenus, benefices, bilans des entreprises\n"
            "- **Indicateurs economiques** : PIB, inflation, taux directeur\n"
            "- **Strategies d'investissement** : analyse de portefeuille, risque\n\n"
            "Posez-moi une question pour commencer !"
        )
    elif lang == "Arabic":
        return (
            "! Ahlan\n\n"
            "I am your financial analysis assistant. I can help you with:\n\n"
            "- **Live market data** from Casablanca Stock Exchange\n"
            "- **Company financials**: revenue, earnings, balance sheets\n"
            "- **Economic indicators**: GDP, inflation, interest rates\n"
            "- **Investment strategies**: portfolio analysis, risk assessment\n\n"
            "Ask me anything about finance to get started!"
        )
    else:
        return (
            "Hello! I'm your AI financial analysis assistant. "
            "I can help you with:\n\n"
            "- **Live market data** from Casablanca Stock Exchange (MASI, all 83 stocks)\n"
            "- **Company financials**: revenue, earnings, balance sheets, segments\n"
            "- **Economic indicators**: GDP, inflation, interest rates, fiscal policy\n"
            "- **Investment strategies**: portfolio analysis, risk assessment\n\n"
            "Ask me anything about finance or the Moroccan market to get started!"
        )


def _is_obviously_unsafe(question_lower: str) -> bool:
    """Detect malicious intent regardless of finance vocabulary used."""
    return any(pattern in question_lower for pattern in _UNSAFE_INTENT_PATTERNS)


def guardrail_check(state: AgentState) -> dict:
    """Node: Fast safety check before hitting retrieval or generation.
    
    SPEED: Uses keyword pre-screening to skip the LLM call for obviously
    safe financial questions (covers ~90% of normal usage).

    PRIORITY OF CHECKS:
      1. Hard refusal on explicit unsafe intent (tax evasion, fraud, etc.)
      2. Fast-pass on safe financial keywords (~90% of normal queries)
      3. LLM safety check for everything else
    """
    question = state["question"].lower()

    # Greetings and conversational queries are always safe
    if _is_greeting(question):
        print("[Guardrail] Fast-pass: greeting/conversational query detected.")
        return {"is_safe": "yes", "is_greeting": True}


    # â›” Hard refusal: explicit malicious intent overrides any safe keyword
    if _is_obviously_unsafe(question):
        print("[Guardrail] Hard refusal: explicit unsafe intent detected.")
        return {"is_safe": "no"}

    # âš¡ Fast path: if question contains known-safe financial keywords, skip LLM call
    if any(kw in question for kw in _OBVIOUSLY_SAFE_PATTERNS):
        print("[Guardrail] Fast-pass: question contains safe financial keywords.")
        return {"is_safe": "yes"}

    print("[Guardrail] Running LLM safety check...")
    llm = _get_fast_llm()
    prompt = GUARDRAIL_PROMPT.format(question=state["question"])

    try:
        res = llm.invoke([SystemMessage(content=prompt)])
        score = res.content.strip().lower()
        is_safe = "no" if "unsafe" in score else "yes"
    except Exception as e:
        print(f"[Guardrail] Error: {e}. Defaulting to safe.")
        is_safe = "yes"

    print(f"[Guardrail] Is safe? {is_safe}")
    return {"is_safe": is_safe}

def generate_blocked(state: AgentState) -> dict:
    """Node: Returns standardized blocked message for unsafe queries.

    The message adapts to the detected language (FR / AR / EN) so the
    user gets a coherent refusal rather than a generic English fallback.
    """
    question = state.get("question", "")
    lang = _detect_language(question)

    if lang == "Arabic":
        blocked_msg = (
            "â›” Ù„Ø§ ÙŠÙ…ÙƒÙ†Ù†ÙŠ Ø§Ù„Ø¥Ø¬Ø§Ø¨Ø© Ø¹Ù„Ù‰ Ù‡Ø°Ø§ Ø§Ù„Ø³Ø¤Ø§Ù„. "
            "Ø¨ØµÙØªÙŠ Ù…Ø³Ø§Ø¹Ø¯Ù‹Ø§ Ù„Ù„ØªØ­Ù„ÙŠÙ„ Ø§Ù„Ù…Ø§Ù„ÙŠØŒ Ù„Ø§ ÙŠÙ…ÙƒÙ†Ù†ÙŠ ØªÙ‚Ø¯ÙŠÙ… Ù…Ø¹Ù„ÙˆÙ…Ø§Øª Ø­ÙˆÙ„ "
            "Ø£Ù†Ø´Ø·Ø© ØºÙŠØ± Ù‚Ø§Ù†ÙˆÙ†ÙŠØ© Ø£Ùˆ Ù…Ø®Ø§Ù„ÙØ© Ù„Ù„Ø£Ø®Ù„Ø§Ù‚ÙŠØ§Øª (Ø§Ù„ØªÙ‡Ø±Ø¨ Ø§Ù„Ø¶Ø±ÙŠØ¨ÙŠØŒ "
            "ØºØ³ÙŠÙ„ Ø§Ù„Ø£Ù…ÙˆØ§Ù„ØŒ Ø§Ù„ØªØ¯Ø§ÙˆÙ„ Ù…Ù† Ø§Ù„Ø¯Ø§Ø®Ù„ØŒ Ø§Ù„Ø§Ø­ØªÙŠØ§Ù„ØŒ Ø¥Ù„Ø®). "
            "ÙŠÙ…ÙƒÙ†Ù†ÙŠ Ù…Ø³Ø§Ø¹Ø¯ØªÙƒ ÙÙŠ ÙÙ‡Ù… Ø§Ù„Ù‚ÙˆØ§Ù†ÙŠÙ† Ø§Ù„Ø¶Ø±ÙŠØ¨ÙŠØ© Ø£Ùˆ Ø§Ù„Ø§Ø³ØªØ«Ù…Ø§Ø± Ø§Ù„Ù…Ø´Ø±ÙˆØ¹."
        )
    elif lang == "French":
        blocked_msg = (
            "â›” Je ne peux pas rÃ©pondre Ã  cette question. En tant "
            "qu'assistant d'analyse financiÃ¨re, je ne peux pas fournir "
            "d'informations sur des activitÃ©s illÃ©gales ou contraires "
            "Ã  l'Ã©thique (Ã©vasion fiscale, blanchiment, dÃ©lit d'initiÃ©, "
            "fraude, etc.). Je peux en revanche vous aider Ã  comprendre "
            "le cadre fiscal lÃ©gal ou les bonnes pratiques d'investissement."
        )
    else:
        blocked_msg = (
            "â›” I cannot answer this question. As a financial analysis "
            "assistant, I cannot provide information on illegal or "
            "unethical activities (tax evasion, money laundering, insider "
            "trading, fraud, etc.). I can however help you understand the "
            "legal tax framework or sound investment practices."
        )

    return {"answer": blocked_msg, "messages": [AIMessage(content=blocked_msg)]}

# â”€â”€ Conversational Context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def contextualize_query(state: AgentState) -> dict:
    """Node: Contextualizes the query using the conversation history.
    
    SPEED: Skipped entirely on first turn (no history).
    Only runs when there are prior conversation turns to resolve pronouns/references.
    """
    history = state.get("history", [])
    question = state["question"]
    
    # âš¡ Fast path: no history â†’ no LLM call needed
    if not history:
        return {"question": question}
    
    # Only bother contextualizing if the question seems to reference prior context
    _ref_words = ["it", "this", "that", "they", "them", "he", "she",
                  "il", "elle", "ce", "cet", "cette", "ils", "elles",
                  "Ù‡Ø°Ø§", "Ù‡Ø°Ù‡", "Ù‡Ù…", "Ù‡ÙŠ", "Ø°Ù„Ùƒ"]
    q_lower = question.lower()
    has_reference = any(f" {w} " in f" {q_lower} " or q_lower.startswith(f"{w} ") for w in _ref_words)
    if not has_reference and len(question.split()) > 4:
        # Long enough standalone question with no pronouns â€” skip the LLM call
        print("[Contextualizer] Skipped (question appears self-contained).")
        return {"question": question}
        
    print("[Contextualizer] Contextualizing query with history...")
    llm = _get_fast_llm()
    
    # Format history into a string (last 4 messages = 2 conversation turns)
    historical_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-4:]])
    
    prompt = CONTEXTUALIZE_PROMPT.format(history=historical_text, question=question)
    
    try:
        res = llm.invoke([SystemMessage(content=prompt)])
        new_q = res.content.strip()
        if new_q.startswith("Standalone Question:"):
            new_q = new_q.split("Standalone Question:")[1].strip()
        new_q = new_q.strip('"').strip("'")
        print(f"[Contextualizer] Contextualized to: {new_q}")
        return {"question": new_q}
    except Exception as e:
        print(f"[Contextualizer] Error: {e}. Keeping original.")
        return {"question": question}

# â”€â”€ CRAG Reflection Nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def grade_documents(state: AgentState) -> dict:
    """Node: Evaluate if the retrieved context is relevant.
    
    Uses a fast heuristic: if context is empty or just "No relevant documents found.",
    trigger the web search fallback. Otherwise, trust the Jina reranker's pre-grading
    (avoids the old problem of the 8B LLM grader being too aggressive on French/Arabic PDFs).
    """
    context = state.get("context", "").strip()
    
    # Empty or no-results context â†’ trigger CRAG fallback to web search
    if not context or context == "No relevant documents found." or len(context) < 50:
        print("[Grader] Context is empty/insufficient â€” marking for web fallback.")
        return {"context_relevant": "no", "retry_count": state.get("retry_count", 0)}
    
    print("[Grader] Context looks sufficient (pre-graded by Jina reranker).")
    return {"context_relevant": "yes", "retry_count": state.get("retry_count", 0)}


def rewrite_query(state: AgentState) -> dict:
    """Node: Rewrite the user query for better retrieval."""
    print("[Rewriter] Rewriting query...")
    llm = _get_fast_llm()
    prompt = REWRITE_PROMPT.format(question=state["question"])
    
    try:
        res = llm.invoke([SystemMessage(content=prompt)])
        new_q = res.content.strip()
        # Extract only the first line â€” LLMs sometimes add explanations after
        new_q = new_q.split("\n")[0].strip().strip('"').strip("'")
        # Hard cap at 300 chars (Tavily limit is 400)
        new_q = new_q[:300] if len(new_q) > 300 else new_q
        if not new_q:
            new_q = state["question"]
    except Exception as e:
        print(f"[Rewriter] Error: {e}. Keeping original.")
        new_q = state["question"]

    print(f"[Rewriter] Old: {state['question'][:80]} -> New: {new_q}")
    return {"question": new_q}


def web_search_fallback(state: AgentState) -> dict:
    """Node: Fallback Web Search using the rewritten query.
    
    Supplements rather than replaces existing context â€” if vector search
    found something partial, we keep it and add web results on top.
    """
    q = state["question"]
    print(f"[Fallback] Web searching for: {q}")
    
    # Use generic search (not financial_news) to avoid biasing non-stock queries
    web_results = _get_web_search().search(q, max_results=5)
    
    # Merge: keep existing context + append web results
    existing_context = state.get("context", "")
    web_context = _get_merger().merge([], [], web_results)
    
    if existing_context and existing_context != "No relevant documents found." and len(existing_context) > 50:
        combined = existing_context + "\n\n---\n\n## Web Search Results\n\n" + web_context
        route = state.get("route", "") + " + Web Search"
    else:
        combined = web_context
        route = "Web Search"
    
    return {
        "context": combined,
        "route": route
    }


# â”€â”€ Build Graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_check)
    graph.add_node("generate_blocked", generate_blocked)
    graph.add_node("contextualize_query", contextualize_query)
    graph.add_node("retrieve",  retrieve_context)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search_fallback", web_search_fallback)
    graph.add_node("generate",  generate_answer)

    graph.set_entry_point("guardrail")
    
    # Guardrail logic
    graph.add_conditional_edges(
        "guardrail",
        lambda state: "contextualize_query" if state.get("is_safe", "yes") == "yes" else "blocked",
        {"contextualize_query": "contextualize_query", "blocked": "generate_blocked"}
    )
    graph.add_edge("generate_blocked", END)
    
    # Contextualizer to Retrieve
    graph.add_edge("contextualize_query", "retrieve")
    
    # Retrieve -> Grade (Reflection Loop)
    graph.add_edge("retrieve", "grade_documents")
    
    # Conditional logic based on grade
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: "generate" if state.get("context_relevant") == "yes" else "rewrite_query",
        {"generate": "generate", "rewrite_query": "rewrite_query"}
    )
    
    # Rewrite -> Web Fallback -> Generate
    graph.add_edge("rewrite_query", "web_search_fallback")
    graph.add_edge("web_search_fallback", "generate")
    
    graph.add_edge("generate", END)

    return graph.compile()


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask(question: str, history: List[BaseMessage] | None = None,
        top_n: int = 5, persona: str = "Balanced",
        focus_mode: str = "New Chat", collection: str = "all") -> dict:
    """
    Ask the financial agent a question.

    Args:
        question: The user's question.
        history:  Optional list of prior BaseMessage objects (for multi-turn memory).
        top_n:    Number of top retrieval results to use.
        persona:  Investment persona (e.g. Balanced, Conservative, Aggressive).
        focus_mode: Focus mode (e.g. New Chat, Company Analysis, Market Macro).
        collection: Target collection (all, financial_data, user_data).

    Returns:
        dict with 'answer' (str), 'context' (str), 'route' (str), and 'history' (list)
    """
    agent = get_agent()
    state = agent.invoke({
        "messages"         : [],
        "history"          : history or [],
        "question"         : question,
        "original_question": question,
        "context"          : "",
        "answer"           : "",
        "route"            : "",
        "context_relevant" : "",
        "is_safe"          : "yes",
        "retry_count"      : 0,
        "top_n"            : top_n,
        "persona"          : persona,
        "focus_mode"       : focus_mode,
        "collection"       : collection,
    })
    # Build updated history for the caller to persist
    updated_history = list(history or []) + [
        HumanMessage(content=question),
        AIMessage(content=state.get("answer", "")),
    ]
    return {
        "answer"  : state.get("answer", ""),
        "context" : state.get("context", ""),
        "route"   : state.get("route", ""),
        "history" : updated_history,
    }


if __name__ == "__main__":
    print("Financial Agent ready. Type your question (Ctrl+C to exit):\n")
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            result = ask(q)
            print(f"\nAdvisor: {result['answer']}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
