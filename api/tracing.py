"""
api/tracing.py
OpenTelemetry tracing utilities for Arize Phoenix.

Provides manual span creation for the full RAG pipeline so that Phoenix
displays a rich trace per user session, not just bare ChatOpenAI LLM calls.

Each query produces a top-level span with user metadata, and child spans for
every pipeline stage: guardrail, contextualize, retrieve, grade, rewrite,
web_fallback, generate.

Usage in routes:
    from api.tracing import trace_query_span, trace_stage

    with trace_query_span(user, request) as span:
        with trace_stage("guardrail") as g:
            ...
            g.set_attribute("guardrail.is_safe", "yes")
        with trace_stage("retrieve") as r:
            ...
            r.set_attribute("retrieve.route", "Vector + Web")
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Optional

from opentelemetry import trace

# The tracer name appears in Phoenix under "Instrumentation Library"
_TRACER_NAME = "financial-advisor.rag-pipeline"


def get_tracer() -> trace.Tracer:
    """Return the application tracer (uses the globally set TracerProvider)."""
    return trace.get_tracer(_TRACER_NAME, "1.0.0")


# ── Semantic Convention Keys ──────────────────────────────────────────────────
# Phoenix / OpenInference recognizes these attribute names and displays them
# in dedicated columns in the UI. See:
# https://docs.arize.com/phoenix/concepts/llm-traces#semantic-attributes

# Session-level (appear on root span)
SESSION_ID = "session.id"
USER_ID = "user.id"
USER_EMAIL = "user.email"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

# OpenInference LLM semantic attributes Phoenix recognizes
OPENINFERENCE_INPUT = "input.value"
OPENINFERENCE_OUTPUT = "output.value"
OPENINFERENCE_METADATA = "metadata"


@contextmanager
def trace_query_span(
    user: dict,
    question: str,
    persona: str = "Balanced",
    focus_mode: str = "New Chat",
    collection: str = "all",
    conversation_id: Optional[str] = None,
):
    """
    Context manager that creates the top-level trace span for a user query.

    This becomes the root span in Phoenix, with user metadata as attributes.
    All pipeline stages traced inside this context become child spans.

    Args:
        user:            Session payload dict (sub, email, role)
        question:        The user's original natural-language question
        persona:         Investment persona setting
        focus_mode:      Focus tab (New Chat / Company Analysis / Market Macro)
        collection:      Target knowledge base collection
        conversation_id: If resuming a conversation
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name="RAG Query",
        kind=trace.SpanKind.SERVER,
    ) as span:
        # ── User & session metadata (displayed in Phoenix columns) ──
        span.set_attribute(USER_ID, user.get("sub", "anonymous"))
        span.set_attribute(USER_EMAIL, user.get("email", "unknown"))
        span.set_attribute(SESSION_ID, conversation_id or f"user-{user.get('sub', '?')}")

        # ── Input / config ──
        span.set_attribute(INPUT_VALUE, question)
        span.set_attribute("query.persona", persona)
        span.set_attribute("query.focus_mode", focus_mode)
        span.set_attribute("query.collection", collection)
        span.set_attribute("query.timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            yield span
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc)[:500])
            span.record_exception(exc)
            raise


@contextmanager
def trace_stage(
    name: str,
    attributes: Optional[dict] = None,
):
    """
    Context manager for tracing an individual pipeline stage as a child span.

    Args:
        name:       Stage name (e.g. "guardrail", "retrieve", "generate")
        attributes: Optional dict of initial attributes to set on the span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name=name,
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, _safe_attr(v))
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc)[:500])
            span.record_exception(exc)
            raise


def set_output(span: trace.Span, output: str) -> None:
    """Set the output value on a span (shown in Phoenix output column)."""
    span.set_attribute(OUTPUT_VALUE, output[:4000] if output else "")


def _safe_attr(v: Any) -> Any:
    """Coerce a value to an OTel-compatible attribute type."""
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return str(v)[:2000]
    return str(v)[:2000]
