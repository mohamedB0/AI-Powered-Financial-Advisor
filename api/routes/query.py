"""
api/routes/query.py
POST /query — accepts a natural language financial question and returns AI insight.
GET  /query/stream — same, but streams the LLM answer token-by-token (SSE).
"""

import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter(prefix="/query", tags=["Query"])


class HistoryMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=15, ge=1, le=30)
    history: Optional[List[HistoryMessage]] = []


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str
    route: str = ""


def _build_history(history: Optional[List[HistoryMessage]]):
    """Convert JSON-serializable history dicts to LangChain BaseMessage objects."""
    if not history:
        return []
    messages = []
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

_semantic_cache = None
def _get_cache():
    global _semantic_cache
    if _semantic_cache is None:
        from retrieval.semantic_cache import SemanticCache
        _semantic_cache = SemanticCache()
    return _semantic_cache


@router.post("/")
def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a financial question and receive an AI-generated insight
    backed by hybrid retrieval (documents + structured data).

    The LangGraph agent runs synchronously — using plain `def` so that
    FastAPI executes it in a thread-pool and the event loop is not blocked.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # 1. Check Semantic Cache
        cache = _get_cache()
        cached = cache.get(request.question)
        if cached:
            return QueryResponse(
                question=request.question,
                answer="🟢 **[Semantic Cache Hit]**\n\n" + cached["answer"],
                context=cached["context"],
                route="cache"
            )

        from agents.financial_agent import ask
        lc_history = _build_history(request.history)
        result = ask(request.question, history=lc_history)
        
        # 2. Update Cache
        if result.get("answer"):
            cache.set(
                query=request.question,
                answer=result["answer"],
                context=result.get("context", ""),
                route=result.get("route", "")
            )

        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            context=result.get("context", ""),
            route=result.get("route", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
def query_stream(request: QueryRequest):
    """
    Same as POST /query but streams the final LLM answer token-by-token.

    The retrieval phase (routing, vector/SQL/web fetch, grading) runs
    synchronously first, then the generation is streamed via SSE.

    Response: text/event-stream — each chunk is a JSON object:
        {"token": "..."} during generation
        {"done": true, "context": "...", "route": "..."} when finished
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_generator():
        import os
        import time
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from agents.financial_agent import (
            retrieve_context, grade_documents, rewrite_query,
            web_search_fallback, AgentState, _get_llm,
        )
        from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE
        from retrieval.query_router import QueryRoute

        # ── 0. Check Semantic Cache (DISABLED) ──────────────────────────
        # cache = _get_cache()
        # cached = cache.get(request.question)
        # if cached:
        #     yield f"data: {{ \"token\": \"🟢 **[Semantic Cache Hit]**\\n\\n\" }}\n\n"
        #     yield f"data: {{ \"token\": {json.dumps(cached['answer'])} }}\n\n"
        #     yield f"data: {{ \"done\": true, \"context\": {json.dumps(cached['context'])}, \"route\": \"cache\" }}\n\n"
        #     return
            
        # ── 1. Retrieval phase (synchronous) ────────────────────────────
        lc_history = _build_history(request.history)
        state: AgentState = {
            "messages"       : [],
            "history"        : lc_history,
            "question"       : request.question,
            "context"        : "",
            "answer"         : "",
            "route"          : "",
            "context_relevant": "",
            "retry_count"    : 0,
        }
        state.update(retrieve_context(state))
        grade = grade_documents(state)
        state.update(grade)

        # CRAG: one rewrite attempt if context is not relevant
        if state.get("context_relevant") == "no":
            state.update(rewrite_query(state))
            state.update(web_search_fallback(state))

        context = state.get("context", "") or "No context available."
        route   = state.get("route", "")

        # ── 2. Streaming generation ─────────────────────────────────────
        llm = _get_llm()
        prior = lc_history[-6:]
        prompt = QUERY_PROMPT_TEMPLATE.format(
            question=request.question,
            context=context,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *prior,
            HumanMessage(content=prompt),
        ]

        try:
            full_answer = ""
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Cache the newly generated answer (DISABLED)
            # if full_answer:
            #     cache.set(
            #         query=request.question,
            #         answer=full_answer,
            #         context=context,
            #         route=route
            #     )

        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Error: {str(e)}]'})}\n\n"

        # ── 3. Done signal with metadata ────────────────────────────────
        yield f"data: {json.dumps({'done': True, 'context': context, 'route': route})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
