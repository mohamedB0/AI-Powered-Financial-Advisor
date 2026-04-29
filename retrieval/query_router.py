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
            model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
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


SQL_KEYWORDS = [
    r"\bprice\b", r"\bstock\b", r"\breturn\b", r"\bclose\b", r"\bopening\b",
    r"\brevenue\b", r"\bebitda\b", r"\bnet.?income\b", r"\beps\b", r"\bearnings\b",
    r"\bgdp\b", r"\binflation\b", r"\bunemployment\b", r"\bfiscal\b", r"\bquarterly\b",
    r"\bvolume\b", r"\bhigh\b", r"\blow\b", r"\bopen\b", r"\bmarket.?cap\b",
    r"\b(p/e|pe.?ratio)\b", r"\bdividend\b", r"\byield\b",
    r"\b(aapl|msft|amzn|googl|meta|tsla|nvda|apple|microsoft|amazon|google|tesla|nvidia)\b",
    r"\bmarket.?mover", r"\btop.?perform", r"\bbest.?stock",
]

WEB_SEARCH_SIGNALS = [
    r"\bnews\b", r"\blatest\b", r"\btoday\b", r"\brecent\b",
    r"\bwhat.?happened\b", r"\bevents?\b", r"\bmarket.?news\b",
    r"\bmerger\b", r"\bacquisition\b", r"\bearnings.?call\b",
    r"\bforecast\b", r"\bprediction\b", r"\b展望\b", r"\b预测\b",
    r"\bhow.?is.*doing\b", r"\bperformance\b", r"\btrending\b",
    r"\bsec.?filing\b", r"\b10-[kq]\b", r"\bquarter.*result\b",
    r"\bwhat's happening\b", r"\bgoing on\b",
    r"\b(morocco|maroc|casa|casablanca)\b", r"\bbvc\b",
    r"\b(attijariwafa|banque populaire|cih|maroc telecom)\b",
]

VECTOR_SIGNALS = [
    # English document signals
    r"\buploaded\b", r"\bfile\b", r"\bpdf\b", r"\bdocument\b",
    r"\breport\b", r"\bpresentation\b", r"\bexcel\b", r"\bcsv\b",
    r"\bfrom.*doc", r"\bdata.*from\b", r"\bin.*doc",
    # French financial terms (Moroccan public finance)
    r"\bmonnaie\b", r"\bdette\b", r"\bbudget\b", r"\bcomptes\b",
    r"\bbilan\b", r"\brecettes\b", r"\bd\u00e9penses\b", r"\bfiscal\b",
    r"\bconjoncture\b", r"\bcroissance\b", r"\br\u00e9cession\b",
    r"\btaux\b", r"\bd\u00e9ficit\b", r"\bexc\u00e9dent\b",
    r"\bcour.?des.?comptes\b", r"\bhcp\b", r"\bbkam\b", r"\bammc\b",
    r"\bbank.?al.?maghrib\b", r"\bautorit\u00e9.?marocain\b",
    r"\br\u00e9sultats\b", r"\br\u00e9sultats\b", r"\br\u00e9sum\u00e9\b",
    r"\brapport\b", r"\bannuel\b", r"\bstat\b", r"\bstatistique\b",
    r"\bpib\b", r"\b\u00e9conomique\b", r"\bpr\u00e9vision\b",
    r"\bplan\b", r"\bprogramme\b", r"\binvestissement\b",
    r"\bsecteur\b", r"\bindustrie\b", r"\bcommerce\b",
    r"\bexportation\b", r"\bimportation\b", r"\bbalance\b",
    r"\bmarocain\b", r"\bmaroc\b", r"\bnational\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pat, text.lower()) for pat in patterns)


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


def _is_bvc_ticker(ticker: Optional[str]) -> bool:
    return ticker is not None and ticker.startswith("BVC.")


def _is_supported_ticker(query: str) -> Tuple[bool, Optional[str]]:
    """Check if query mentions a ticker we actually support in SQL."""
    ticker = _extract_ticker(query)
    if ticker is None:
        return False, None
    if ticker.startswith("BVC."):
        return False, ticker
    return True, ticker


def _region_is_unsupported(query: str) -> bool:
    """Check if query mentions a region we don't have data for in SQL."""
    q = query.lower()
    unsupported_regions = [
        "morocco", "maroc", "casa", "casablanca", "bvc",
        "moroccan", "البورصة", "الدار البيضاء",
        "attijariwafa", "banque populaire", "cih", "maroc telecom",
        "attijari", "wafa", "bank al maghrib", "al morocco"
    ]
    return any(region in q for region in unsupported_regions)


def _classify_intent(query: str) -> Tuple[QueryRoute, Optional[str]]:
    """
    Classify query intent and determine the primary route.
    
    Priority order:
      1. Unsupported region / BVC ticker → WEB (hard override)
      2. Web signals (news, events, live) → WEB
      3. SQL-only (financial data, no vector signals) → SQL
      4. Vector-only (documents, no SQL signals) → VECTOR
      5. Both SQL + vector → HYBRID
      6. No clear signal → LLM fallback → VECTOR (default)
    
    Returns:
        Tuple of (route, detail) where:
        - route: Primary QueryRoute
        - detail: Optional hint (e.g., ticker symbol)
    """
    q = query.lower()

    sql_match = _matches_any(query, SQL_KEYWORDS)
    web_match = _matches_any(query, WEB_SEARCH_SIGNALS)
    vector_match = _matches_any(query, VECTOR_SIGNALS)
    unsupported_region = _region_is_unsupported(query)

    has_supported_ticker, extracted_ticker = _is_supported_ticker(query)
    is_bvc_ticker = extracted_ticker is not None and extracted_ticker.startswith("BVC.")

    # 1. Hard override: unsupported region or BVC ticker → always WEB
    if unsupported_region or is_bvc_ticker:
        return QueryRoute.WEB, None

    # 2. Web signals take priority for news/event queries
    if web_match:
        return QueryRoute.WEB, None

    # 3. Pure SQL queries (financial data, metrics, prices) — only if no vector overlap
    if sql_match and not vector_match:
        ticker = _extract_ticker(query)
        return QueryRoute.SQL, ticker

    # 4. Pure vector/document queries
    if vector_match and not sql_match:
        return QueryRoute.VECTOR, None

    # 5. Hybrid: both SQL and vector signals present
    if sql_match:
        ticker = _extract_ticker(query)
        return QueryRoute.HYBRID, ticker
    
    # 6. LLM Fallback: no clear regex signal
    try:
        from agents.prompts import ROUTER_PROMPT
        from langchain_core.messages import HumanMessage
        llm = _get_router_llm()
        resp = llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(question=query))])
        label = resp.content.strip().lower().split()[0]  # first word only
        if label in ("sql", "vector", "web", "hybrid"):
            route = QueryRoute(label)
            ticker = _extract_ticker(query) if label in ("sql", "hybrid") else None
            print(f"[Router] LLM fallback classified as: {label}")
            return route, ticker
    except Exception as e:
        print(f"[Router] LLM fallback error: {e}. Defaulting to VECTOR.")
    # Final fallback
    return QueryRoute.VECTOR, None


def classify_query(query: str) -> dict:
    """
    Classify a query and return routing information.
    
    Args:
        query: The user's question
        
    Returns:
        dict with:
            - route: QueryRoute enum value
            - detail: Optional routing hint
            - sql_keywords: List of matched SQL keywords
            - web_keywords: List of matched web search keywords
            - reason: Human-readable explanation
    """
    route, detail = _classify_intent(query)
    
    matched_sql = [p for p in SQL_KEYWORDS if re.search(p, query.lower())]
    matched_web = [p for p in WEB_SEARCH_SIGNALS if re.search(p, query.lower())]
    matched_vector = [p for p in VECTOR_SIGNALS if re.search(p, query.lower())]
    
    reason_map = {
        QueryRoute.SQL: "Query matched SQL/structured data keywords (stocks, financials, macro)",
        QueryRoute.WEB: "Query matched web search signals or is a general question",
        QueryRoute.VECTOR: "Query is document/knowledge-based",
        QueryRoute.HYBRID: "Query matches both SQL and vector patterns - using hybrid approach",
    }
    
    return {
        "route": route.value,
        "detail": detail,
        "sql_keywords": matched_sql[:5],
        "web_keywords": matched_web[:5],
        "vector_keywords": matched_vector[:5],
        "reason": reason_map.get(route, "Default routing"),
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
