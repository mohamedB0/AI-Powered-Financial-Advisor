"""
agents/multi_agent.py
Plan-and-Execute Multi-Agent Architecture for the Financial Advisor.

Architecture:
  1. PLANNER AGENT   — Decomposes complex questions into sub-tasks
  2. EXECUTOR AGENTS — Specialist agents that each handle one sub-task:
     - SQL Executor:    Structured data (stock prices, metrics, BVC)
     - RAG Executor:    Document retrieval (AMMC, BKAM, HCP reports)
     - Web Executor:    Current events, news, live data
  3. SYNTHESIZER AGENT — Merges all sub-task results into a final coherent answer

This builds on top of the existing retrieval infrastructure (VectorSearch, SQLSearch,
WebSearch, QueryRouter) but orchestrates them via a multi-agent LangGraph workflow
instead of a single monolithic agent.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_planner_llm():
    """Strong LLM for planning (needs good reasoning)."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.1, max_tokens=1024, max_retries=2, request_timeout=20,
        )
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1, max_tokens=1024, max_retries=2, request_timeout=30,
    )


def _get_executor_llm():
    """Fast LLM for executor sub-tasks (speed matters)."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.0, max_tokens=512, max_retries=2, request_timeout=10,
        )
    return ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0, max_tokens=512, max_retries=2, request_timeout=20,
    )


def _get_synthesizer_llm():
    """Strong LLM for synthesis (needs coherent multilingual output)."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0.3, max_tokens=2048, max_retries=2, request_timeout=25,
            model_kwargs={"frequency_penalty": 0.4, "presence_penalty": 0.2},
        )
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3, max_tokens=2048, max_retries=2, request_timeout=40,
        model_kwargs={"frequency_penalty": 0.4, "presence_penalty": 0.2},
    )


# ══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class SubTask(BaseModel):
    """A single sub-task produced by the Planner."""
    id: int = Field(description="Task number (1-based)")
    description: str = Field(description="What to do")
    agent: str = Field(description="Which executor: 'sql', 'rag', or 'web'")
    query: str = Field(description="The specific query for the executor")


class MultiAgentState(TypedDict):
    """Shared state for the Plan-and-Execute multi-agent graph."""
    # Input
    question: str
    original_question: str
    history: List[BaseMessage]
    collection: str

    # Guardrail output
    is_safe: str               # "yes" or "no"
    blocked_reason: str        # Why the query was blocked (empty if safe)

    # Planner output
    plan: List[dict]           # List of sub-task dicts
    is_simple: bool            # If True, skip planning and go direct

    # Executor outputs
    sql_result: str
    rag_result: str
    web_result: str

    # Synthesizer output
    answer: str
    route: str


# ══════════════════════════════════════════════════════════════════════════════
# NODE 0: GUARDRAIL AGENT
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that are ALWAYS safe (financial domain keywords)
_SAFE_PATTERNS = [
    # English
    "stock", "price", "market", "revenue", "earnings", "income", "ebitda",
    "eps", "dividend", "bond", "equity", "gdp", "inflation", "interest rate",
    "portfolio", "investment", "balance sheet", "cash flow", "valuation",
    "financial", "profit", "margin", "debt", "asset",
    # French
    "bourse", "action", "marché", "taux", "obligation", "investissement",
    "rendement", "portefeuille", "croissance", "dividende", "budget",
    "dette", "fiscal", "monétaire", "économie", "PIB",
    # Arabic
    "سهم", "أسهم", "سوق", "تضخم", "فائدة", "استثمار", "عائد", "أرباح",
    "شركة", "بورصة", "اقتصاد", "بنك", "مصرف", "قرض", "ميزانية", "مالي",
    # Moroccan institutions
    "ammc", "bkam", "hcp", "bank al-maghrib", "cour des comptes",
    "attijariwafa", "bcp", "iam", "managem", "cosumar", "ocp",
]

# Patterns that trigger HARD REFUSAL (illegal intent)
_UNSAFE_PATTERNS = [
    # English
    "how to evade", "tax evasion", "money laundering", "launder money",
    "insider trading", "manipulate the market", "pump and dump",
    "ponzi", "scam", "fraud", "embezzle", "bribe",
    "hide money", "hide income",
    # French
    "évasion fiscale", "fraude fiscale", "blanchiment", "blanchir",
    "délit d'initié", "manipuler le marché", "cacher de l'argent",
    "comment frauder", "détournement", "pot-de-vin", "corruption",
    # Arabic
    "التهرب الضريبي", "غسيل الأموال", "غسل الأموال", "الاحتيال",
    "إخفاء الأموال", "التداول من الداخل", "التلاعب بالسوق",
    "اختلاس", "رشوة", "نصب", "تبييض الاموال",
]

# Blocked response templates per language
_BLOCKED_RESPONSES = {
    "Arabic": (
        "لا يمكنني الإجابة على هذا السؤال. "
        "بصفتي مساعدا للتحليل المالي، لا يمكنني تقديم معلومات حول "
        "أنشطة غير قانونية أو مخالفة للأخلاقيات. "
        "يمكنني مساعدتك في فهم القوانين المالية أو الاستثمار المشروع."
    ),
    "French": (
        "Je ne peux pas repondre a cette question. En tant "
        "qu'assistant d'analyse financiere, je ne peux pas fournir "
        "d'informations sur des activites illegales ou contraires "
        "a l'ethique (evasion fiscale, blanchiment, delit d'initie, "
        "fraude, etc.). Je peux en revanche vous aider a comprendre "
        "le cadre fiscal legal ou les bonnes pratiques d'investissement."
    ),
    "English": (
        "I cannot answer this question. As a financial analysis "
        "assistant, I cannot provide information on illegal or "
        "unethical activities (tax evasion, money laundering, insider "
        "trading, fraud, etc.). I can however help you understand the "
        "legal tax framework or sound investment practices."
    ),
}


def _detect_language(text: str) -> str:
    """Detect French/Arabic/English from text."""
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "Arabic"
    french_words = {"le", "la", "les", "des", "est", "que", "qui", "dans",
                    "pour", "sur", "pas", "une", "avec", "quel", "quelle",
                    "comment", "pourquoi", "quelles", "sont", "cette"}
    words = set(text.lower().split())
    if len(words & french_words) >= 2:
        return "French"
    return "English"


def guardrail(state: MultiAgentState) -> dict:
    """
    Guardrail Agent: Multi-layer safety check before any execution.

    Layer 1: Hard refusal on explicit unsafe intent (instant, no LLM)
    Layer 2: Fast-pass on known safe financial keywords (instant, no LLM)
    Layer 3: LLM-based classification for ambiguous queries
    """
    question = state["question"]
    q_lower = question.lower()

    print(f"\n[GUARDRAIL] Checking: {question[:60]}...")

    # Layer 1: Hard refusal — explicit illegal intent
    for pattern in _UNSAFE_PATTERNS:
        if pattern in q_lower:
            lang = _detect_language(question)
            blocked_msg = _BLOCKED_RESPONSES[lang]
            print(f"[GUARDRAIL] BLOCKED — matched unsafe pattern: '{pattern}'")
            return {
                "is_safe": "no",
                "blocked_reason": f"Unsafe intent detected: {pattern}",
                "answer": blocked_msg,
                "route": "Blocked (Guardrail)",
            }

    # Layer 2: Fast-pass — known safe financial keywords
    if any(kw in q_lower for kw in _SAFE_PATTERNS):
        print("[GUARDRAIL] PASSED (safe financial keywords detected)")
        return {"is_safe": "yes", "blocked_reason": ""}

    # Layer 3: LLM-based check for ambiguous queries
    print("[GUARDRAIL] Ambiguous query — running LLM safety check...")
    llm = _get_executor_llm()

    guardrail_prompt = (
        "You are a security guardrail for a financial AI assistant.\n"
        "Classify if this question is SAFE or UNSAFE.\n\n"
        "UNSAFE means: asking for illegal activity (tax evasion, fraud, "
        "money laundering, insider trading, market manipulation) or "
        "completely unrelated to finance/economics.\n\n"
        "SAFE means: any legitimate financial, economic, market, investment, "
        "or business question, even if speculative or complex.\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY one word: safe or unsafe"
    )

    try:
        response = llm.invoke([SystemMessage(content=guardrail_prompt)])
        verdict = response.content.strip().lower()
        is_safe = "no" if "unsafe" in verdict else "yes"
    except Exception as e:
        print(f"[GUARDRAIL] LLM error: {e}. Defaulting to safe.")
        is_safe = "yes"

    if is_safe == "no":
        lang = _detect_language(question)
        blocked_msg = _BLOCKED_RESPONSES[lang]
        print(f"[GUARDRAIL] BLOCKED by LLM classification")
        return {
            "is_safe": "no",
            "blocked_reason": "LLM classified as unsafe",
            "answer": blocked_msg,
            "route": "Blocked (Guardrail)",
        }

    print(f"[GUARDRAIL] PASSED (LLM confirmed safe)")
    return {"is_safe": "yes", "blocked_reason": ""}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: PLANNER AGENT
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_PROMPT = """You are a financial query planner. Your job is to decompose the user's question
into 1-3 sub-tasks, each assigned to a specialist executor agent.

Available executor agents:
- "sql": Handles structured data — stock prices, financial metrics (P/E, EPS, revenue),
  company fundamentals, BVC (Moroccan stock exchange) data. Use when the question asks
  for specific numbers, prices, or comparisons between companies.
- "rag": Handles document knowledge — Moroccan institutional reports (AMMC, Bank Al-Maghrib,
  HCP, Cour des Comptes), policy documents, regulations, macroeconomic analysis papers.
  Use when the question asks about policies, regulations, reports, or concepts.
- "web": Handles current events — today's news, recent market movements, live data,
  forecasts, breaking events. Use when the question needs information newer than
  the knowledge base.

Rules:
1. Output ONLY valid JSON — an array of objects with keys: id, description, agent, query.
2. Use 1 sub-task for simple questions, 2-3 for complex multi-part questions.
3. Each sub-task query should be self-contained (the executor won't see other tasks).
4. Keep the user's language in the query field (French stays French, Arabic stays Arabic).
5. Never create more than 3 sub-tasks.

Examples:

Question: "What is the price of AAPL?"
[{{"id": 1, "description": "Get AAPL stock price", "agent": "sql", "query": "AAPL stock price latest"}}]

Question: "Compare Attijariwafa's stock performance with Bank Al-Maghrib's monetary policy impact"
[{{"id": 1, "description": "Get ATW stock data", "agent": "sql", "query": "Attijariwafa ATW stock price and metrics"}},
 {{"id": 2, "description": "Find BKAM monetary policy documents", "agent": "rag", "query": "politique monetaire Bank Al-Maghrib impact"}}]

Question: "Quelles sont les dernières nouvelles sur la bourse de Casablanca et quels sont les cours actuels?"
[{{"id": 1, "description": "Get live BVC stock prices", "agent": "sql", "query": "BVC cotation cours actions Casablanca"}},
 {{"id": 2, "description": "Search recent BVC news", "agent": "web", "query": "bourse Casablanca dernières nouvelles aujourd'hui"}}]

User question: {question}
"""


def plan(state: MultiAgentState) -> dict:
    """Planner Agent: decomposes the question into sub-tasks."""
    question = state["question"]
    print(f"\n[PLANNER] Analyzing: {question[:80]}...")

    llm = _get_planner_llm()
    prompt = PLANNER_PROMPT.format(question=question)

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        text = response.content.strip()

        # Parse JSON from the response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        tasks = json.loads(text)

        # Validate structure
        validated = []
        for t in tasks[:3]:  # Max 3 tasks
            validated.append({
                "id": t.get("id", len(validated) + 1),
                "description": t.get("description", ""),
                "agent": t.get("agent", "rag"),
                "query": t.get("query", question),
            })

        is_simple = len(validated) <= 1
        print(f"[PLANNER] Created {len(validated)} sub-task(s): {[t['agent'] for t in validated]}")
        for t in validated:
            print(f"  Task {t['id']} [{t['agent'].upper()}]: {t['description']}")

        return {"plan": validated, "is_simple": is_simple}

    except Exception as e:
        print(f"[PLANNER] Error: {e}. Falling back to single RAG task.")
        fallback = [{"id": 1, "description": "Answer the question", "agent": "rag", "query": question}]
        return {"plan": fallback, "is_simple": True}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: EXECUTOR AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def execute_sql(state: MultiAgentState) -> dict:
    """SQL Executor Agent: handles structured data queries."""
    tasks = [t for t in state["plan"] if t["agent"] == "sql"]
    if not tasks:
        return {"sql_result": ""}

    print(f"\n[SQL EXECUTOR] Processing {len(tasks)} task(s)...")
    from retrieval.sql_search import SQLSearch
    from retrieval.query_router import _extract_ticker

    sql = SQLSearch()
    results_parts = []

    for task in tasks:
        query = task["query"]
        q_lower = query.lower()
        print(f"  Executing: {task['description']}")

        try:
            # Try to extract a ticker from the query
            ticker = _extract_ticker(query)

            if ticker:
                # Get price data for specific ticker
                price_data = sql.get_price_history(ticker, days=10)
                latest = sql.get_latest_price(ticker)
                if latest:
                    results_parts.append(f"**{ticker} - Latest Price:**\n{sql.format_results(latest)}")
                if price_data:
                    results_parts.append(f"**{ticker} - 10-Day History:**\n{sql.format_results(price_data)}")

                # Also try financial metrics
                metrics = sql.get_financial_metrics(ticker)
                if metrics:
                    results_parts.append(f"**{ticker} - Financial Metrics:**\n{sql.format_results(metrics)}")

            elif any(k in q_lower for k in ["revenue", "earnings", "top", "meilleur"]):
                top = sql.get_top_by_revenue()
                if top:
                    results_parts.append(f"**Top Companies by Revenue:**\n{sql.format_results(top)}")

            elif any(k in q_lower for k in ["mover", "perform", "gainer", "loser"]):
                movers = sql.get_top_movers()
                if movers:
                    results_parts.append(f"**Market Movers:**\n{sql.format_results(movers)}")

            elif any(k in q_lower for k in ["volume", "active", "actif"]):
                vol = sql.get_highest_volume()
                if vol:
                    results_parts.append(f"**Most Active (Volume):**\n{sql.format_results(vol)}")

            else:
                # Generic: try BVC prices
                bvc = sql.get_bvc_latest_prices()
                if bvc:
                    results_parts.append(f"**BVC Latest Prices:**\n{sql.format_results(bvc[:10])}")

        except Exception as e:
            print(f"  SQL error: {e}")
            results_parts.append(f"SQL query failed for: {task['description']}")

    result = "\n\n".join(results_parts) if results_parts else "No structured data found."
    print(f"[SQL EXECUTOR] Done. Result length: {len(result)} chars")
    return {"sql_result": result}


def execute_rag(state: MultiAgentState) -> dict:
    """RAG Executor Agent: handles document retrieval queries."""
    tasks = [t for t in state["plan"] if t["agent"] == "rag"]
    if not tasks:
        return {"rag_result": ""}

    print(f"\n[RAG EXECUTOR] Processing {len(tasks)} task(s)...")
    from retrieval.vector_search import VectorSearch

    vs = VectorSearch()
    results_parts = []

    collection = state.get("collection", "all")
    coll_filter = None if collection == "all" else collection

    for task in tasks:
        print(f"  Searching: {task['description']}")
        try:
            results = vs.search(task["query"], top_k=5, collection=coll_filter)
            if results:
                formatted = vs.format_context(results)
                results_parts.append(formatted)
            else:
                results_parts.append(f"No documents found for: {task['query']}")
        except Exception as e:
            print(f"  RAG error: {e}")
            results_parts.append(f"Document search failed for: {task['description']}")

    result = "\n\n---\n\n".join(results_parts) if results_parts else "No relevant documents found."
    print(f"[RAG EXECUTOR] Done. Result length: {len(result)} chars")
    return {"rag_result": result}


def execute_web(state: MultiAgentState) -> dict:
    """Web Executor Agent: handles current events and live data queries."""
    tasks = [t for t in state["plan"] if t["agent"] == "web"]
    if not tasks:
        return {"web_result": ""}

    print(f"\n[WEB EXECUTOR] Processing {len(tasks)} task(s)...")
    from retrieval.web_search import WebSearch

    web = WebSearch()
    results_parts = []

    for task in tasks:
        print(f"  Searching web: {task['description']}")
        try:
            results = web.search(task["query"], max_results=5)
            if results:
                formatted = web.format_results(results)
                results_parts.append(formatted)
            else:
                results_parts.append(f"No web results for: {task['query']}")
        except Exception as e:
            print(f"  Web error: {e}")
            results_parts.append(f"Web search failed for: {task['description']}")

    result = "\n\n".join(results_parts) if results_parts else "No web results found."
    print(f"[WEB EXECUTOR] Done. Result length: {len(result)} chars")
    return {"web_result": result}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: SYNTHESIZER AGENT
# ══════════════════════════════════════════════════════════════════════════════

SYNTHESIZER_PROMPT = """You are an expert AI Financial Advisor. You receive results from multiple
specialist agents (SQL database, document search, web search) and must synthesize them into
a single coherent answer for the user.

## Rules
1. Answer the user's question DIRECTLY. Start with the key answer.
2. Use ONLY information from the provided agent results. Never invent data.
3. If multiple agents provide relevant information, integrate them seamlessly.
4. If some agent results are empty or irrelevant, ignore them gracefully.
5. CRITICAL: Reply in the SAME LANGUAGE as the user's question.
6. Use markdown formatting (bold, bullet points) for readability. No emojis.
7. Never mention internal systems, databases, agents, or retrieval methods.
8. Be concise but thorough. Every sentence must be relevant.

## Agent Results

### SQL Agent (Structured Data)
{sql_result}

### Document Agent (Knowledge Base)
{rag_result}

### Web Agent (Current Events)
{web_result}

## User Question
{question}

## Your Answer:"""


def synthesize(state: MultiAgentState) -> dict:
    """Synthesizer Agent: merges all executor results into a final answer."""
    print(f"\n[SYNTHESIZER] Merging results from executors...")

    sql_result = state.get("sql_result", "") or "No structured data available."
    rag_result = state.get("rag_result", "") or "No documents found."
    web_result = state.get("web_result", "") or "No web results."

    # Determine which agents contributed
    agents_used = []
    if sql_result and sql_result != "No structured data available." and sql_result != "":
        agents_used.append("SQL")
    if rag_result and rag_result != "No relevant documents found." and rag_result != "No documents found.":
        agents_used.append("RAG")
    if web_result and web_result != "No web results found." and web_result != "No web results.":
        agents_used.append("Web")

    route_label = " + ".join(agents_used) if agents_used else "No Sources"

    # Build the synthesis prompt
    prompt = SYNTHESIZER_PROMPT.format(
        sql_result=sql_result,
        rag_result=rag_result,
        web_result=web_result,
        question=state["original_question"],
    )

    llm = _get_synthesizer_llm()

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert financial advisor. Answer precisely and concisely."),
            HumanMessage(content=prompt),
        ])
        answer = response.content.strip()
        print(f"[SYNTHESIZER] Done. Answer length: {len(answer)} chars")
    except Exception as e:
        print(f"[SYNTHESIZER] Error: {e}")
        # Fallback: return raw results
        answer = (
            f"Here are the findings from the analysis:\n\n"
            f"**Structured Data:**\n{sql_result}\n\n"
            f"**Documents:**\n{rag_result}\n\n"
            f"**Web Search:**\n{web_result}"
        )

    return {"answer": answer, "route": f"Multi-Agent ({route_label})"}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_plan(state: MultiAgentState) -> str:
    """Determine which executors to run based on the plan."""
    agents_needed = set(t["agent"] for t in state.get("plan", []))

    # Run executors in parallel by going to a dispatch node
    return "execute"


def build_multi_agent():
    """Build the Plan-and-Execute multi-agent LangGraph."""
    graph = StateGraph(MultiAgentState)

    # Add nodes
    graph.add_node("guardrail", guardrail)
    graph.add_node("planner", plan)
    graph.add_node("sql_executor", execute_sql)
    graph.add_node("rag_executor", execute_rag)
    graph.add_node("web_executor", execute_web)
    graph.add_node("synthesizer", synthesize)

    # Entry point: Guardrail checks safety first
    graph.set_entry_point("guardrail")

    # Conditional: if safe -> planner, if blocked -> END
    graph.add_conditional_edges(
        "guardrail",
        lambda state: "planner" if state.get("is_safe") == "yes" else "end",
        {"planner": "planner", "end": END},
    )

    # After planning, run executors sequentially (they self-filter based on plan)
    graph.add_edge("planner", "sql_executor")
    graph.add_edge("sql_executor", "rag_executor")
    graph.add_edge("rag_executor", "web_executor")
    graph.add_edge("web_executor", "synthesizer")

    # Synthesizer is the final node
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_multi_agent = None


def get_multi_agent():
    """Get or create the multi-agent graph singleton."""
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = build_multi_agent()
    return _multi_agent


def ask_multi_agent(
    question: str,
    history: List[BaseMessage] | None = None,
    collection: str = "all",
) -> dict:
    """
    Ask the Plan-and-Execute multi-agent system a question.

    Args:
        question:   The user's question (French, Arabic, or English).
        history:    Optional conversation history.
        collection: Target collection ("all", "financial_data", "user_data").

    Returns:
        dict with 'answer', 'route', 'plan', and 'history'.
    """
    agent = get_multi_agent()

    state = agent.invoke({
        "question": question,
        "original_question": question,
        "history": history or [],
        "collection": collection,
        "is_safe": "",
        "blocked_reason": "",
        "plan": [],
        "is_simple": False,
        "sql_result": "",
        "rag_result": "",
        "web_result": "",
        "answer": "",
        "route": "",
    })

    # Build updated history
    updated_history = list(history or []) + [
        HumanMessage(content=question),
        AIMessage(content=state.get("answer", "")),
    ]

    return {
        "answer": state.get("answer", ""),
        "route": state.get("route", ""),
        "plan": state.get("plan", []),
        "history": updated_history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MULTI-AGENT FINANCIAL ADVISOR (Plan-and-Execute)")
    print("=" * 60)
    print("Type your question (Ctrl+C to exit):\n")

    history = []
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            result = ask_multi_agent(q, history=history)
            history = result["history"]
            print(f"\nRoute: {result['route']}")
            print(f"Plan: {[t['agent'] for t in result['plan']]}")
            print(f"\nAdvisor: {result['answer']}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
