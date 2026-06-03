"""
agents/financial_agent.py
LangGraph-based financial analysis agent.

Uses:
  - OpenRouter API for LLM inference (meta-llama/llama-3.1-8b-instruct)
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
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from retrieval.query_router import route_query, QueryRoute
from retrieval.vector_search import VectorSearch
from retrieval.sql_search import SQLSearch
from retrieval.web_search import WebSearch
from retrieval.hybrid_merger import HybridMerger
from retrieval.reranker import Reranker
from retrieval.bm25_search import BM25Search
from retrieval.rrf_fusion import reciprocal_rank_fusion
from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE, GRADER_PROMPT, REWRITE_PROMPT, GUARDRAIL_PROMPT, CONTEXTUALIZE_PROMPT
from agents.tools import TOOLS


# ── Agent State ───────────────────────────────────────────────────────────────

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


# ── LLM Setup ─────────────────────────────────────────────────────────────────

# ── Speed: Groq is ~10-20x faster than OpenRouter for equivalent models ───────
# Primary path: Groq (ultra-fast inference)
# Fallback path: OpenRouter (if Groq key missing or quota exceeded)

_GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Fastest high-quality Groq model
    "llama-3.1-8b-instant",       # Ultra-fast fallback
]

_FALLBACK_MODELS = [
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct:free",
]

def _get_llm(model: str | None = None):
    """Returns the best available LLM. Prefers Groq for speed."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and model is None:
        # Use Groq primary model — significantly faster inference
        groq_model = os.getenv("GROQ_MODEL", _GROQ_MODELS[0])
        return ChatOpenAI(
            model=groq_model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.3, max_tokens=1024, max_retries=0, request_timeout=30,
        )
    # OpenRouter fallback (or explicit model override)
    if model is None:
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    return ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3, max_tokens=1024, max_retries=0, request_timeout=60,
    )

def _get_fast_llm():
    """Returns a fast/cheap LLM for guardrails, routing, rewriting etc. Prefers Groq."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.0, max_tokens=256, max_retries=0, request_timeout=10,
        )
    model = os.getenv("FAST_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
    return ChatOpenAI(
        model=model, base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0, max_tokens=256, max_retries=0, request_timeout=15,
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# ── Lazy-loaded retrieval singletons (avoid 25s+ init at import time) ─────────

_vector_search = None
_sql_search = None
_web_search = None
_merger = None

def _get_vector_search():
    global _vector_search
    if _vector_search is None:
        _vector_search = VectorSearch()
    return _vector_search

def _get_sql_search():
    global _sql_search
    if _sql_search is None:
        _sql_search = SQLSearch()
    return _sql_search

def _get_web_search():
    global _web_search
    if _web_search is None:
        _web_search = WebSearch()
    return _web_search

def _get_merger():
    global _merger
    if _merger is None:
        _merger = HybridMerger()
    return _merger

_reranker = None
def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

_bm25_search = None
def _get_bm25_search():
    global _bm25_search
    if _bm25_search is None:
        _bm25_search = BM25Search()
    return _bm25_search


UNSUPPORTED_REGIONS: list = []  # All regions supported; BVC tickers are in the DB
US_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]


def _expand_query(question: str) -> str:
    """
    Generate an alternative search query to improve recall.
    Uses the Fast LLM to rephrase the question with different keywords.
    """
    try:
        from langchain_core.messages import SystemMessage
        llm = _get_fast_llm()
        prompt = (
            "Generate ONE alternative search query for the following question. "
            "Use different keywords and synonyms but keep the same intent. "
            "Output ONLY the alternative query, nothing else.\n\n"
            f"Original: {question}\nAlternative:"
        )
        res = llm.invoke([SystemMessage(content=prompt)])
        alt = res.content.strip().split("\n")[0].strip().strip('"').strip("'")
        return alt[:200] if alt else ""
    except Exception:
        return ""


def _results_match_query(results: list, question: str) -> bool:
    """Check if SQL results actually match the query (ticker AND data type)."""
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

    if "atw" in q or "attijari" in q:
        return any("ATW" in str(r) or "BVC.ATW" in str(r) for r in results)
    if "bcm" in q or "bcp" in q or "banque populaire" in q:
        return any("BCP" in str(r) or "BVC.BCP" in str(r) for r in results)
    if "iam" in q or "maroc telecom" in q:
        return any("IAM" in str(r) or "BVC.IAM" in str(r) for r in results)
    for ticker in US_TICKERS:
        if ticker.lower() in q:
            return any(ticker in str(r) for r in results)
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

    # ── 1. SQL Search ────────────────────────────────────────────────────────
    if need_sql:
        try:
            if routing["detail"]:
                sql_results = _get_sql_search().get_price_history(routing["detail"], days=10)
            elif "revenue" in q_lower or "earnings" in q_lower:
                sql_results = _get_sql_search().get_top_by_revenue()
            elif "volume" in q_lower:
                sql_results = _get_sql_search().get_highest_volume()
            else:
                sql_results = _get_sql_search().get_top_movers()

            # Validate: do SQL results actually match what the user asked?
            if sql_results and not _results_match_query(sql_results, question):
                print(f"[Router] SQL results don't match query — discarding")
                sql_results = []
        except Exception as e:
            print(f"[Router] SQL error: {e}")

    # ── 2. Hybrid Retrieval: Dense (BGE-M3) + Sparse (BM25) + RRF + Reranker ──
    vector_quality = "none"  # Track: "none", "weak", "strong"
    if need_vector:
        # Determine target collection(s) based on user selection
        target_collection = state.get("collection", "all")
        coll_filter = None if target_collection == "all" else target_collection

        try:
            # 2a. Dense retrieval (semantic embeddings via ChromaDB) — wider pool for better recall
            dense_results = _get_vector_search().search(question, top_k=30, collection=coll_filter)
            # Tag results for RRF source tracking
            for r in dense_results:
                r["retrieval_method"] = "dense"

            # 2b. Sparse retrieval (BM25 keyword matching) — wider pool
            bm25_results = []
            try:
                bm25_results = _get_bm25_search().search(question, top_k=30)
            except Exception as e:
                print(f"[Router] BM25 search error (non-fatal): {e}")

            # 2c. Query expansion — generate an alternative query for better recall
            expanded_results = []
            try:
                alt_query = _expand_query(question)
                if alt_query and alt_query.lower() != question.lower():
                    print(f"[Router] Query expansion: '{alt_query[:60]}'")
                    expanded_dense = _get_vector_search().search(alt_query, top_k=15, collection=coll_filter)
                    for r in expanded_dense:
                        r["retrieval_method"] = "dense_expanded"
                    expanded_results = expanded_dense
            except Exception as e:
                print(f"[Router] Query expansion error (non-fatal): {e}")

            # 2d. Reciprocal Rank Fusion — merge all retrieval streams
            all_streams = [s for s in [dense_results, bm25_results, expanded_results] if s]
            if all_streams:
                if len(all_streams) > 1:
                    total_docs = sum(len(s) for s in all_streams)
                    print(f"[Router] Hybrid retrieval: {total_docs} total docs from {len(all_streams)} streams -> RRF merge")
                    vector_results = reciprocal_rank_fusion(*all_streams, top_k=25)
                else:
                    vector_results = all_streams[0]

            # 2e. Rerank fused results with Jina cross-encoder
            if vector_results:
                dynamic_top_n = state.get("top_n", 5)
                print(f"[Router] Reranking {len(vector_results)} hybrid results via Jina API (top_n={dynamic_top_n})...")
                vector_results = _get_reranker().rerank(question, vector_results, top_n=dynamic_top_n)

                # 2f. Filter out low-confidence results (improves Context Precision)
                if vector_results:
                    min_score = 0.25
                    before_filter = len(vector_results)
                    vector_results = [r for r in vector_results if r.get("rerank_score", 0) >= min_score]
                    if len(vector_results) < before_filter:
                        print(f"[Router] Filtered {before_filter - len(vector_results)} low-confidence results (< {min_score})")

                # Check reranker confidence
                if vector_results:
                    best_score = max(r.get("rerank_score", 0) for r in vector_results)
                    print(f"[Router] Best reranker score: {best_score:.4f} ({len(vector_results)} docs)")
                    vector_quality = "strong" if best_score >= 0.3 else "weak"

        except Exception as e:
            print(f"[Router] Hybrid retrieval error: {e}")

    # ── 3. Web Search ────────────────────────────────────────────────────
    #   Always runs if:
    #     a) Router explicitly asked for web  OR
    #     b) Vector results are weak/absent (auto-supplement to avoid "je n'ai pas trouvé")
    if need_web or vector_quality in ("none", "weak"):
        if vector_quality in ("none", "weak") and not need_web:
            print(f"[Router] Vector quality '{vector_quality}' — auto-supplementing with web search.")
        try:
            web_results = _get_web_search().search(question, max_results=3)
        except Exception as e:
            print(f"[Router] Web search error: {e}")

    # ── 4. Merge only the sources that were fetched ──────────────────────
    # If vector quality is weak but web returned results, drop weak vectors to avoid confusion
    if vector_quality == "weak" and web_results:
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
    lang_instruction = f"\n\n⚠️ MANDATORY: Your entire response MUST be written in {detected_lang}. The user asked in {detected_lang}, so reply ONLY in {detected_lang}."
    prompt += lang_instruction

    # Prepend prior conversation history (last 4 messages = 2 turns) to save tokens
    prior = state.get("history", [])[-4:]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *prior,
        HumanMessage(content=prompt),
    ]

    # ⚡ Speed: Try Groq first (None → uses Groq if key present), then OpenRouter fallbacks.
    # Passing None lets _get_llm() pick Groq automatically.
    openrouter_primary = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    # (None sentinel = Groq preferred path)
    models_to_try = [None] + [m for m in _FALLBACK_MODELS if m != openrouter_primary] + [openrouter_primary]

    for model_name in models_to_try:
        current_llm = _get_llm(model_name)
        # Retry with exponential backoff on rate limit (up to 2 attempts per model)
        for attempt in range(2):
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
                    if attempt == 0:
                        wait = 2
                        print(f"[LLM] Rate limit on {model_name}. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"[LLM] Rate limit exhausted for {model_name}. Trying next model.")
                    break  # try next model

                if "provider" in error_msg.lower() or "502" in error_msg or "503" in error_msg:
                    print(f"[LLM] Provider error on {model_name}: {error_msg[:120]}. Trying next model.")
                    break  # try next model

                if "tool_use_failed" in error_msg or "failed_generation" in error_msg:
                    return {
                        "messages": [],
                        "answer": "I encountered an issue processing your request. Please try rephrasing your question.",
                    }

                # Other unexpected error — log and try next model
                print(f"[LLM] Unexpected error on {model_name} attempt {attempt + 1}: {error_msg[:120]}")
                if attempt == 0:
                    time.sleep(1)
                    continue
                break  # try next model

    # All models exhausted
    fallback = (
        "**⚠️ Service Temporarily Unavailable**\n\n"
        "The AI provider is currently experiencing issues. "
        "Please try again in a moment.\n\n"
        "Here is the raw evidence retrieved from your documents:\n\n"
        f"{state['context'] or 'No context available.'}"
    )
    return {"messages": [], "answer": fallback}


def should_use_tools(state: AgentState) -> str:
    """Edge: decide whether to call a tool or end."""
    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ── Guardrail Node ────────────────────────────────────────────────────────────

_OBVIOUSLY_SAFE_PATTERNS = [
    # Finance / econ terms in French, English, Arabic
    "marché", "bourse", "action", "obligation", "inflation", "taux", "pib", "croissance",
    "dividende", "rendement", "risque", "portefeuille", "investissement", "bénéfice",
    "stock", "market", "price", "revenue", "gdp", "interest", "bond", "equity",
    "سهم", "سوق", "تضخم", "فائدة", "ناتج", "استثمار", "عائد",
    "budget", "dette", "fiscal", "hcp", "bank al-maghrib", "ammc", "bkam",
]

def guardrail_check(state: AgentState) -> dict:
    """Node: Fast safety check before hitting retrieval or generation.
    
    SPEED: Uses keyword pre-screening to skip the LLM call for obviously
    safe financial questions (covers ~90% of normal usage).
    """
    question = state["question"].lower()
    
    # ⚡ Fast path: if question contains known-safe financial keywords, skip LLM call
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
    """Node: Returns standardized blocked message for unsafe queries."""
    blocked_msg = "⛔ **Guardrail Alert**: As an AI Financial Advisor, I cannot provide direct guaranteed investment advice, illegal financial information, or discuss non-financial topics."
    return {"answer": blocked_msg, "messages": [AIMessage(content=blocked_msg)]}

# ── Conversational Context ────────────────────────────────────────────────────

def contextualize_query(state: AgentState) -> dict:
    """Node: Contextualizes the query using the conversation history.
    
    SPEED: Skipped entirely on first turn (no history).
    Only runs when there are prior conversation turns to resolve pronouns/references.
    """
    history = state.get("history", [])
    question = state["question"]
    
    # ⚡ Fast path: no history → no LLM call needed
    if not history:
        return {"question": question}
    
    # Only bother contextualizing if the question seems to reference prior context
    _ref_words = ["it", "this", "that", "they", "them", "he", "she",
                  "il", "elle", "ce", "cet", "cette", "ils", "elles",
                  "هذا", "هذه", "هم", "هي", "ذلك"]
    q_lower = question.lower()
    has_reference = any(f" {w} " in f" {q_lower} " or q_lower.startswith(f"{w} ") for w in _ref_words)
    if not has_reference and len(question.split()) > 4:
        # Long enough standalone question with no pronouns — skip the LLM call
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

# ── CRAG Reflection Nodes ─────────────────────────────────────────────────────

def grade_documents(state: AgentState) -> dict:
    """Node: Evaluate if the retrieved context is relevant.
    
    Uses a fast heuristic: if context is empty or just "No relevant documents found.",
    trigger the web search fallback. Otherwise, trust the Jina reranker's pre-grading
    (avoids the old problem of the 8B LLM grader being too aggressive on French/Arabic PDFs).
    """
    context = state.get("context", "").strip()
    
    # Empty or no-results context → trigger CRAG fallback to web search
    if not context or context == "No relevant documents found." or len(context) < 50:
        print("[Grader] Context is empty/insufficient — marking for web fallback.")
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
        # Extract only the first line — LLMs sometimes add explanations after
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
    
    Supplements rather than replaces existing context — if vector search
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


# ── Build Graph ───────────────────────────────────────────────────────────────

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
    graph.add_node("tools",     ToolNode(TOOLS))

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
    
    graph.add_conditional_edges("generate", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "generate")

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask(question: str, history: List[BaseMessage] | None = None) -> dict:
    """
    Ask the financial agent a question.

    Args:
        question: The user's question.
        history:  Optional list of prior BaseMessage objects (for multi-turn memory).

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
        "is_safe"          : "yes",
        "retry_count"      : 0,
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
