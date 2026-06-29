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


@router.get("/collections")
def list_collections() -> dict:
    """
    Return the knowledge-base collections available for querying, with a live
    document count for each. Used by the chat UI to render the collection
    selector menu bar.

    Returns:
        {
          "active": "all",
          "collections": [
            {"id": "all",            "label": "All Sources",    "count": 12345},
            {"id": "financial_data", "label": "Institutional",  "count": 12000},
            {"id": "user_data",      "label": "My Uploads",     "count": 345}
          ]
        }
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from storage.chroma_store import (
        CHROMA_PERSIST_DIR,
        CHROMA_COLLECTION_FINANCIAL,
        CHROMA_COLLECTION_USER_DATA,
    )

    counts = {CHROMA_COLLECTION_FINANCIAL: 0, CHROMA_COLLECTION_USER_DATA: 0}
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        for name in counts:
            try:
                coll = client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
                counts[name] = coll.count()
            except Exception:
                counts[name] = 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read collections: {e}")

    financial_n = counts[CHROMA_COLLECTION_FINANCIAL]
    user_n = counts[CHROMA_COLLECTION_USER_DATA]

    return {
        "active": "all",
        "collections": [
            {
                "id": "all",
                "label": "All Sources",
                "description": "Institutional reports + your uploads",
                "icon": "fa-layer-group",
                "count": financial_n + user_n,
            },
            {
                "id": CHROMA_COLLECTION_FINANCIAL,
                "label": "Institutional Docs",
                "description": "Scraped reports (AMMC, BKAM, HCP, CDC)",
                "icon": "fa-building-columns",
                "count": financial_n,
            },
            {
                "id": CHROMA_COLLECTION_USER_DATA,
                "label": "My Uploads",
                "description": "Documents you ingested",
                "icon": "fa-folder-open",
                "count": user_n,
            },
        ],
    }


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
        from api.tracing import trace_query_span, trace_stage, set_output

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

        with trace_query_span(
            user=user,
            question=request.question,
            persona=state["persona"],
            focus_mode=state["focus_mode"],
            collection=state["collection"],
        ) as root_span:

            # ── 1. Contextualize + Guardrail + Retrieve (parallel) ────────
            with trace_stage("contextualize") as ctx_span:
                state.update(contextualize_query(state))
                ctx_span.set_attribute("contextualize.output_question", state["question"][:300])

            # Run guardrail and retrieval in parallel threads, then record spans
            # after the results are back (avoids OTel context detach errors from
            # cross-thread span management).
            import time as _time
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                t0 = _time.time()
                future_guardrail = executor.submit(guardrail_check, state)
                future_retrieval = executor.submit(retrieve_context, state)
                retrieval_result = future_retrieval.result()
                guardrail_result = future_guardrail.result()

            # Record retrieval span (after thread completed — no cross-thread context)
            with trace_stage("retrieve") as ret_span:
                ret_span.set_attribute("retrieve.route", retrieval_result.get("route", ""))
                ret_span.set_attribute("retrieve.context_length", len(retrieval_result.get("context", "")))

            # Record guardrail span
            with trace_stage("guardrail") as guard_span:
                guard_span.set_attribute("guardrail.is_safe", guardrail_result.get("is_safe", "yes"))

            # Check safety gate
            if guardrail_result.get("is_safe", "yes") != "yes":
                with trace_stage("generate_blocked") as blk_span:
                    blocked = generate_blocked(state)
                    blk_span.set_attribute("blocked", True)
                root_span.set_attribute("query.blocked", True)
                set_output(root_span, blocked["answer"])
                yield f"data: {json.dumps({'token': blocked['answer']})}\n\n"
                yield f"data: {json.dumps({'done': True, 'context': '', 'route': 'blocked'})}\n\n"
                return

            state.update(retrieval_result)
            state.update(guardrail_result)

            # ── 2. Grade & optional CRAG rewrite ──────────────────────────
            with trace_stage("grade_documents") as grade_span:
                grade = grade_documents(state)
                state.update(grade)
                grade_span.set_attribute("grade.context_relevant", state.get("context_relevant", ""))

            if state.get("context_relevant") == "no":
                with trace_stage("rewrite_query") as rw_span:
                    rewrite_result = rewrite_query(state)
                    state.update(rewrite_result)
                    rw_span.set_attribute("rewrite.new_question", state["question"][:300])

                with trace_stage("web_search_fallback") as ws_span:
                    fallback_result = web_search_fallback(state)
                    state.update(fallback_result)
                    ws_span.set_attribute("fallback.route", state.get("route", ""))
                    ws_span.set_attribute("fallback.context_length", len(state.get("context", "")))

            context = state.get("context", "") or "No context available."
            route = state.get("route", "")

            # Record route on root span
            root_span.set_attribute("query.route", route)
            root_span.set_attribute("query.context_length", len(context))

            # ── 3. Streaming generation ───────────────────────────────────
            prior = lc_history[-4:]
            prompt = QUERY_PROMPT_TEMPLATE.format(
                question=request.question,
                context=context,
            )

            from agents.financial_agent import _detect_language
            detected_lang = _detect_language(request.question)
            prompt += f"\n\n⚠️ MANDATORY: Your entire response MUST be written in {detected_lang}. The user asked in {detected_lang}, so reply ONLY in {detected_lang}."

            root_span.set_attribute("query.detected_language", detected_lang)

            persona_guide = f"You must strongly adopt the following Investment Persona: {state.get('persona', 'Balanced')}."

            messages = [
                SystemMessage(content=SYSTEM_PROMPT + "\n\n" + persona_guide),
                *prior,
                HumanMessage(content=prompt),
            ]

            from agents.financial_agent import _FALLBACK_MODELS, _get_llm as _agent_get_llm

            openrouter_primary = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
            models_to_try = [None] + [m for m in _FALLBACK_MODELS if m != openrouter_primary] + [openrouter_primary]

            streamed = False
            with trace_stage("generate") as gen_span:
                for model_name in models_to_try:
                    try:
                        llm_stream = _agent_get_llm(model_name)
                        gen_span.set_attribute("generate.model", model_name or "groq-primary")
                        full_answer = ""
                        _gen_start = _time.time()
                        _STREAM_TIMEOUT = 60  # seconds — hard cutoff for hung LLMs
                        for chunk in llm_stream.stream(messages):
                            # Timeout protection: if no complete response after 60s, bail
                            if _time.time() - _gen_start > _STREAM_TIMEOUT:
                                gen_span.set_attribute("generate.timeout", True)
                                yield f"data: {json.dumps({'token': '\n\n⚠️ **Response timeout.** The model took too long. Please try again.'})}\n\n"
                                break
                            token = chunk.content if hasattr(chunk, "content") else str(chunk)
                            if token:
                                full_answer += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        else:
                            # Loop completed normally (no break)
                            streamed = True
                            gen_span.set_attribute("generate.output_length", len(full_answer))
                            set_output(root_span, full_answer[:2000])
                        break  # success or timeout — exit model loop
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
                            gen_span.set_attribute(f"generate.fallback.{model_name or 'groq'}", error_msg[:100])
                            print(f"[Stream] Provider error on {model_name}: {error_msg[:100]}. Trying next model.")
                            continue
                        err_token = (
                            "\n\n⚠️ **Le service IA est temporairement indisponible.** "
                            "Veuillez réessayer dans quelques instants. "
                            f"*(Détail technique: {error_msg[:120]})*"
                        )
                        gen_span.set_attribute("generate.error", error_msg[:300])
                        yield f"data: {json.dumps({'token': err_token})}\n\n"
                        break

            # ── 4. Done signal with metadata ──────────────────────────────
            yield f"data: {json.dumps({'done': True, 'context': context, 'route': route})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Answer Feedback ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "positive" or "negative"
    timestamp: Optional[str] = None


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    """
    Store user feedback (thumbs up/down) on an answer.
    Used for quality tracking and evaluation.
    """
    import logging
    logger = logging.getLogger("feedback")

    # Log to file for analysis
    logger.info(
        f"[FEEDBACK] user={user.get('email')} rating={req.rating} "
        f"question={req.question[:100]} answer_len={len(req.answer)}"
    )

    # Also trace in Phoenix
    try:
        from api.tracing import trace_stage
        with trace_stage("user_feedback", {
            "feedback.rating": req.rating,
            "feedback.user_email": user.get("email", ""),
            "feedback.question": req.question[:200],
        }):
            pass
    except Exception:
        pass

    return {"status": "ok", "rating": req.rating}
