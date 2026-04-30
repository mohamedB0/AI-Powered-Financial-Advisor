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
from retrieval.web_search import WebSearch
from retrieval.hybrid_merger import HybridMerger
from retrieval.reranker import Reranker
from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE, GRADER_PROMPT, REWRITE_PROMPT, GUARDRAIL_PROMPT
from agents.tools import TOOLS


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages         : Annotated[List[BaseMessage], lambda a, b: a + b]
    history          : List[BaseMessage]          # prior conversation turns
    question         : str
    context          : str
    answer           : str
    route            : str
    context_relevant : str
    is_safe          : str                        # CRAG guard: safety flag
    retry_count      : int                        # CRAG guard: max 2 rewrites


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key,
        temperature=0.1, max_tokens=1024, max_retries=0, request_timeout=60,
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


UNSUPPORTED_REGIONS = ["morocco", "maroc", "casa", "casablanca", "bvc", "moroccan", "البورصة", "الدار البيضاء"]
US_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]


def _results_match_query(results: list, question: str) -> bool:
    """Check if SQL results actually match the query (ticker AND data type AND region detection)."""
    if not results:
        return False
    q = question.lower()
    result_str = str(results).upper()

    is_revenue_query = any(k in q for k in ["revenue", "earning", "income", "ebitda", "eps"])
    is_price_query = any(k in q for k in ["price", "close", "open", "high", "low", "volume", "stock"])
    is_macro_query = any(k in q for k in ["gdp", "inflation", "unemployment", "macro"])

    if is_revenue_query:
        has_financial_data = any(word in result_str for word in ["REVENUE", "NET_INCOME", "EBITDA", "EPS", "FINANCIAL"])
        if not has_financial_data:
            print(f"[Router] Query asks for financial metrics but got stock prices")
            return False
    elif is_macro_query:
        has_macro_data = any(word in result_str for word in ["COUNTRY", "INDICATOR", "MACRO", "GDP", "INFLATION"])
        if not has_macro_data:
            print(f"[Router] Query asks for macro data but got other data")
            return False
    elif is_price_query:
        has_price_data = any(word in result_str for word in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DATE"])
        if not has_price_data:
            print(f"[Router] Query asks for price data but got other data")
            return False

    for region in UNSUPPORTED_REGIONS:
        if region in q:
            print(f"[Router] Query mentions '{region}' but SQL only has US stocks - falling back to web")
            return False

    if "atw" in q or "attijari" in q:
        return any("ATW" in str(r) or "BVC.ATW" in str(r) for r in results)
    if "bcm" in q or "banque populaire" in q:
        return any("BCM" in str(r) or "BVC.BCM" in str(r) for r in results)
    if "iam" in q or "maroc telecom" in q or "ita" in q:
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

    need_sql    = route in ("sql", "hybrid")
    need_vector = route in ("vector", "hybrid")
    need_web    = route in ("web", "hybrid")

    # ── 1. SQL Search ────────────────────────────────────────────────────
    if need_sql:
        try:
            if routing["detail"]:
                sql_results = _get_sql_search().get_price_history(routing["detail"], days=10)
            elif "inflation" in q_lower:
                sql_results = _get_sql_search().get_macro_indicator("United States", "inflation")
            elif "gdp" in q_lower:
                sql_results = _get_sql_search().get_macro_indicator("United States", "gdp")
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

    # ── 2. Vector Search (with Cross-Encoder Reranking) ──────────────────
    if need_vector:
        try:
            # Over-fetch with fast vector search
            vector_results = _get_vector_search().search(question, top_k=30)
            
            # Re-rank with precision cross-encoder if results exist
            if vector_results:
                print(f"[Router] Reranking {len(vector_results)} vector results...")
                vector_results = _get_reranker().rerank(question, vector_results, top_n=15)
        except Exception as e:
            print(f"[Router] Vector/Reranker error: {e}")

    # ── 3. Web Search ────────────────────────────────────────────────────
    if need_web:
        bvc_tickers = ["atw", "bcp", "bcm", "cfg", "cih", "iam", "lhm", "mng", "sam", "slf", "waa", "tqm", "boa", "msa", "adh", "lbv"]
        is_bvc_query = any(t in q_lower for t in bvc_tickers) or any(r in q_lower for r in ["morocco", "maroc", "casablanca", "bvc", "attijariwafa", "attijari", "banque populaire"])

        if is_bvc_query:
            web_results = _get_web_search().get_morocco_market_data(question)
            if not web_results:
                # Append "Morocco Casablanca stock" so DuckDuckGo finds the right company
                morocco_query = f"{question} Morocco Casablanca stock exchange"
                web_results = _get_web_search().search_financial_news(morocco_query, max_results=5)
        else:
            web_results = _get_web_search().search_financial_news(question, max_results=5)

    # ── 4. Merge only the sources that were fetched ──────────────────────
    context = _get_merger().merge(vector_results, sql_results, web_results)

    return {
        "context": context,
        "route": route,
    }


def generate_answer(state: AgentState) -> AgentState:
    """Node: Generate answer using LLM with retrieved context."""
    # Do NOT bind_tools — free OpenRouter models don't support tool calling.
    # Tool execution is handled separately by the ToolNode via should_use_tools edge.
    llm = _get_llm()

    prompt = QUERY_PROMPT_TEMPLATE.format(
        question=state["question"],
        context=state["context"] or "No context available.",
    )

    # Prepend prior conversation history (last 6 messages = 3 turns)
    prior = state.get("history", [])[-6:]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *prior,
        HumanMessage(content=prompt),
    ]

    # Retry with exponential backoff on rate limit (up to 3 attempts)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return {
                "messages": [response],
                "answer"  : response.content,
            }
        except Exception as e:
            error_msg = str(e)

            if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"[LLM] Rate limit hit. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                # All retries exhausted — return context as fallback
                fallback = (
                    "**LLM Rate Limit Reached.**\n\n"
                    "Unable to synthesize an answer right now. "
                    "Here is the raw evidence retrieved from your documents:\n\n"
                    f"{state['context'] or 'No context available.'}"
                )
                return {"messages": [], "answer": fallback}

            if "tool_use_failed" in error_msg or "failed_generation" in error_msg:
                return {
                    "messages": [],
                    "answer": "I encountered an issue processing your request. Please try rephrasing your question.",
                }

            # Catch-all: log the actual error and return context as fallback
            print(f"[LLM] Unexpected error on attempt {attempt + 1}: {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            fallback = (
                f"**LLM Error:** {error_msg[:200]}\n\n"
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

def guardrail_check(state: AgentState) -> dict:
    """Node: Fast safety check before hitting retrieval or generation."""
    print("[Guardrail] Checking safety...")
    llm = _get_llm()
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

# ── CRAG Reflection Nodes ─────────────────────────────────────────────────────

def grade_documents(state: AgentState) -> dict:
    """Node: Evaluate if the retrieved context is relevant."""
    print("[Grader] Evaluating relevance of context...")

    # CRAG guard: if we've already retried twice, stop the loop
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        print("[Grader] Max retries reached — forcing generate.")
        return {"context_relevant": "yes", "retry_count": retry_count}

    if not state["context"] or state["context"] == "No relevant documents found.":
        print("[Grader] Relevance: no")
        return {"context_relevant": "no", "retry_count": retry_count + 1}

    # Skip grader LLM call for vector/sql routes — context is curated and always relevant.
    # This saves one API call per query and avoids hitting rate limits.
    route = state.get("route", "")
    if route in (QueryRoute.VECTOR.value, QueryRoute.SQL.value):
        print(f"[Grader] Skipping grader for {repr(route)} route - assuming relevant.")
        return {"context_relevant": "yes", "retry_count": retry_count}

    # Use plain text generation instead of with_structured_output
    # (free OpenRouter models don't support structured output / tool calling)
    llm = _get_llm()
    prompt = GRADER_PROMPT.format(context=state["context"], question=state["question"])

    try:
        res = llm.invoke([SystemMessage(content=prompt)])
        text = res.content.strip().lower()
        # Parse the yes/no from the plain text response
        score = "yes" if "yes" in text[:50] else "no"
    except Exception as e:
        print(f"[Grader] Error: {e}. Defaulting to relevant.")
        score = "yes"

    new_retry = retry_count + (1 if score == "no" else 0)
    print(f"[Grader] Relevance: {score} (retry_count={new_retry})")
    return {"context_relevant": score, "retry_count": new_retry}


def rewrite_query(state: AgentState) -> dict:
    """Node: Rewrite the user query for better retrieval."""
    print("[Rewriter] Rewriting query...")
    llm = _get_llm()
    prompt = REWRITE_PROMPT.format(question=state["question"])
    
    try:
        res = llm.invoke([SystemMessage(content=prompt)])
        new_q = res.content.strip()
    except Exception as e:
        print(f"[Rewriter] Error: {e}. Keeping original.")
        new_q = state["question"]
        
    print(f"[Rewriter] Old query: {state['question']} -> New query: {new_q}")
    return {"question": new_q}


def web_search_fallback(state: AgentState) -> dict:
    """Node: Fallback Web Search using the rewritten query."""
    q = state["question"]
    print(f"[Fallback] Web searching for: {q}")
    
    web_results = _get_web_search().search_financial_news(q, max_results=5)
    # Provide the Web Search results as the new context
    new_context = _get_merger().merge([], [], web_results)
    
    return {
        "context": new_context,
        "route": "web_fallback"
    }


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_check)
    graph.add_node("generate_blocked", generate_blocked)
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
        lambda state: "retrieve" if state.get("is_safe", "yes") == "yes" else "blocked",
        {"retrieve": "retrieve", "blocked": "generate_blocked"}
    )
    graph.add_edge("generate_blocked", END)
    
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
        "messages"    : [],
        "history"     : history or [],
        "question"    : question,
        "context"     : "",
        "answer"      : "",
        "route"       : "",
        "is_safe"     : "yes",
        "retry_count" : 0,
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
