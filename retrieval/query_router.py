"""
retrieval/query_router.py
Query classification and routing for Hybrid RAG.

Routes queries to:
  1. SQL tools (structured data: stocks, financials, macro)
  2. Web search (current events, news, live data)
  3. Vector search (documents, knowledge base)

When regex signals are ambiguous, a lightweight LLM call is used as a
fallback classifier (lazy-loaded, cached to avoid repeated instantiation).
"""

import os
import re
from enum import Enum
from typing import List, Optional, Tuple

# ── Lazy LLM for fallback classification ──────────────────────────────────────
_router_llm = None

def _get_router_llm():
    global _router_llm
    if _router_llm is None:
        from langchain_openai import ChatOpenAI
        _router_llm = ChatOpenAI(
            model=os.getenv("FAST_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.0,
            max_retries=0,
            request_timeout=15,
        )
    return _router_llm


class QueryRoute(str, Enum):
    SQL = "sql"
    WEB = "web"
    VECTOR = "vector"
    HYBRID = "hybrid"


# ── Ticker Extraction (Standardized helper) ──────────────────────────────
def _extract_ticker(query: str) -> Optional[str]:
    us_tickers = {
        "AAPL": r"\b(aapl|apple)\b", "MSFT": r"\b(msft|microsoft)\b",
        "AMZN": r"\b(amzn|amazon)\b", "GOOGL": r"\b(googl|google|alphabet)\b",
        "META": r"\b(meta|facebook)\b", "TSLA": r"\b(tsla|tesla)\b",
        "NVDA": r"\b(nvda|nvidia)\b",
    }
    bvc_tickers = {
        "ATW": r"\b(atw|attijariwafa|attijari)\b",
        "BCP": r"\b(bcp|bcm|banque\.?populaire|banque\.?centrale\.?populaire)\b",
        "CFG": r"\b(cfg|cfagr?oup|credit\.?financier|cfg\.?bank)\b",
        "CIH": r"\b(cih|cih\.?bank)\b",
        "IAM": r"\b(iam|ita|itissalat|maroc\.?telecom)\b",
        "LHM": r"\b(lhm|lafarge|lafarge\.?holcim\.?maroc)\b",
        "MNG": r"\b(mng|managem)\b",
        "SAM": r"\b(sam|samir|raffinerie)\b",
        "SLF": r"\b(slf|salafin)\b",
        "WAA": r"\b(waa|wav|wafasalaf|wafa\.?assurance)\b",
        "TQM": r"\b(tqm|taqa\.?morocco|taqa)\b",
        "BOA": r"\b(boa|bank\.?of\.?africa|bmce)\b",
        "MSA": r"\b(msa|marsa\.?maroc)\b",
        "ADH": r"\b(adh|addoha)\b",
        "LBV": r"\b(lbv|label\.?vie)\b",
    }
    q = query.lower()
    for ticker, pattern in us_tickers.items():
        if re.search(pattern, q):
            return ticker
    for ticker, pattern in bvc_tickers.items():
        if re.search(pattern, q):
            return f"BVC.{ticker}"
    return None

import json

def _classify_intent_llm(query: str) -> dict:
    """Uses the LLM to structurally understand and route the user's intent."""
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _get_router_llm()
    
    system_prompt = """You are an elite financial RAG router managing requests in English, French, and Arabic.
Analyze the user's question and strictly return ONLY a JSON dictionary deciding which database to query.

ROUTES:
- "sql": ONLY if asking for strict stock market data, market cap, yields, P/E ratios, or exact numeric ticker metrics.
- "vector": IF asking about policies, macroeconomics, inflation, employment, text reports, general facts, AMMC, Bank Al-Maghrib, HCP, or unknown concepts.
- "web": IF asking about literally today's news, latest world events, or sudden future predictions.
- "hybrid": IF asking for both stock market hard numbers AND report literature.

JSON SCHEMA:
{
  "route": "sql" | "vector" | "web" | "hybrid",
  "reason": "A 1-sentence analytical justification"
}"""
    
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        # Clean JSON formatting securely
        text = resp.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        
        valid_routes = [r.value for r in QueryRoute]
        route_str = data.get("route", "vector").lower()
        if route_str not in valid_routes:
            route_str = "vector"
            
        return {
            "route": QueryRoute(route_str),
            "reason": data.get("reason", "Intelligent LLM Default Override")
        }
    except Exception as e:
        print(f"[Router] Semantic routing failed: {e}. Defaulting to VECTOR.")
        return {"route": QueryRoute.VECTOR, "reason": "Fallback due to LLM error"}


def classify_query(query: str) -> dict:
    """
    Classify a query and return routing information via Semantic LLM Routing.
    """
    llm_decision = _classify_intent_llm(query)
    route = llm_decision["route"]
    reason = llm_decision["reason"]
    
    # Retrieve structured ticker cleanly using Regex if going to SQL
    detail = None
    if route in (QueryRoute.SQL, QueryRoute.HYBRID):
        detail = _extract_ticker(query)
        
    return {
        "route": route.value,
        "detail": detail,
        "sql_keywords": [],
        "web_keywords": [],
        "vector_keywords": [],
        "reason": reason,
    }


def get_required_tools(route: QueryRoute) -> List[str]:
    """
    Determine which tools to use based on the route.
    """
    tool_map = {
        QueryRoute.SQL: ["sql"],
        QueryRoute.WEB: ["web_search"],
        QueryRoute.VECTOR: ["vector"],
        QueryRoute.HYBRID: ["sql", "vector"],
    }
    return tool_map.get(route, ["vector"])


def route_query(query: str) -> dict:
    """
    Main routing function - classify and return routing decision.
    """
    classification = classify_query(query)
    route = QueryRoute(classification["route"])
    required_tools = get_required_tools(route)
    
    return {
        **classification,
        "required_tools": required_tools,
        "skip_vector": route == QueryRoute.SQL,
        "skip_sql": route == QueryRoute.VECTOR or route == QueryRoute.WEB,
        "skip_web": route != QueryRoute.WEB and route != QueryRoute.HYBRID,
    }
