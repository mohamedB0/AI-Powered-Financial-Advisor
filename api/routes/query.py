"""
api/routes/query.py
POST /query — accepts a natural language financial question and returns AI insight.
GET  /query/stream — same, but streams the LLM answer token-by-token (SSE).
"""

import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from api.auth import get_current_user

router = APIRouter(prefix="/query", tags=["Query"])


class HistoryMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=30)
    history: Optional[List[HistoryMessage]] = []
    persona: Optional[str] = "Balanced"
    focus_mode: Optional[str] = "New Chat"
    collection: Optional[str] = "all"  # "all", "financial_data", or "user_data"


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
def query(request: QueryRequest, user: dict = Depends(get_current_user)) -> QueryResponse:
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
def query_stream(request: QueryRequest, user: dict = Depends(get_current_user)):
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
        import concurrent.futures
        from langchain_core.messages import SystemMessage, HumanMessage
        from agents.financial_agent import (
            retrieve_context, grade_documents, rewrite_query,
            web_search_fallback, guardrail_check, generate_blocked,
            contextualize_query, AgentState,
        )
        from agents.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE

        lc_history = _build_history(request.history)

        state: AgentState = {
            "messages"         : [],
            "history"          : lc_history,
            "question"         : request.question,
            "original_question": request.question,
            "context"          : "",
            "answer"           : "",
            "route"            : "",
            "context_relevant" : "",
            "is_safe"          : "yes",
            "retry_count"      : 0,
            "persona"          : request.persona or "Balanced",
            "top_n"            : request.top_k or 3,
            "focus_mode"       : request.focus_mode or "New Chat",
            "collection"       : request.collection or "all",
        }

        # ⚡ PARALLEL: Run guardrail check + retrieval at the same time.
        # For ~90% of financial queries the guardrail is a free keyword fast-pass,
        # so retrieval finishes first and we never wait for guardrail at all.
        # For the remaining 10% (LLM call needed), retrieval and guardrail overlap.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Contextualize before retrieval (fast: skips LLM when no history/pronouns)
            state.update(contextualize_query(state))

            future_guardrail = executor.submit(guardrail_check, state)
            future_retrieval  = executor.submit(retrieve_context, state)

            retrieval_result  = future_retrieval.result()   # wait for retrieval
            guardrail_result  = future_guardrail.result()   # wait for guardrail

        # Check safety gate
        if guardrail_result.get("is_safe", "yes") != "yes":
            blocked = generate_blocked(state)
            yield f"data: {json.dumps({'token': blocked['answer']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'context': '', 'route': 'blocked'})}\n\n"
            return

        state.update(retrieval_result)
        state.update(guardrail_result)

        # Grade & optional CRAG rewrite
        grade = grade_documents(state)
        state.update(grade)
        if state.get("context_relevant") == "no":
            state.update(rewrite_query(state))
            state.update(web_search_fallback(state))

        context = state.get("context", "") or "No context available."
        route   = state.get("route", "")

        # ── 2. Streaming generation ─────────────────────────────────────
        prior = lc_history[-4:]
        prompt = QUERY_PROMPT_TEMPLATE.format(
            question=request.question,
            context=context,
        )

        # Enforce response language matches user's question language
        from agents.financial_agent import _detect_language
        detected_lang = _detect_language(request.question)
        prompt += f"\n\n⚠️ MANDATORY: Your entire response MUST be written in {detected_lang}. The user asked in {detected_lang}, so reply ONLY in {detected_lang}."
        
        persona_guide = f"You must strongly adopt the following Investment Persona: {state.get('persona', 'Balanced')}."
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\n" + persona_guide),
            *prior,
            HumanMessage(content=prompt),
        ]

        from agents.financial_agent import _FALLBACK_MODELS, _get_llm as _agent_get_llm

        openrouter_primary = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        # ⚡ None sentinel = Groq preferred path (fastest), then OpenRouter fallbacks
        models_to_try = [None] + [m for m in _FALLBACK_MODELS if m != openrouter_primary] + [openrouter_primary]

        streamed = False
        for model_name in models_to_try:
            try:
                llm_stream = _agent_get_llm(model_name)
                full_answer = ""
                for chunk in llm_stream.stream(messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
                streamed = True
                break  # success — exit model loop
            except Exception as e:
                error_msg = str(e)
                is_provider_error = (
                    "provider" in error_msg.lower()
                    or "502" in error_msg
                    or "503" in error_msg
                    or "rate_limit" in error_msg
                    or "429" in error_msg
                )
                if is_provider_error and model_name != models_to_try[-1]:
                    print(f"[Stream] Provider error on {model_name}: {error_msg[:100]}. Trying next model.")
                    continue  # try next model
                # Final failure — emit a clean user-facing error
                err_token = (
                    "\n\n⚠️ **Le service IA est temporairement indisponible.** "
                    "Veuillez réessayer dans quelques instants. "
                    f"*(Détail technique: {error_msg[:120]})*"
                )
                yield f"data: {json.dumps({'token': err_token})}\n\n"
                break

        # ── 3. Done signal with metadata ────────────────────────────────
        yield f"data: {json.dumps({'done': True, 'context': context, 'route': route})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
