"""
api/routes/vision.py
POST /vision/analyze — Two-step financial image analysis:
  Step 1: Extract text/data from the image using OCR/Vision extraction
  Step 2: Pass extracted content to Groq LLaMA 3.3-70B for financial analysis

This separates concerns:
  - OCR/extraction model handles reading charts, tables, text from images
  - Generation model (same as the main RAG pipeline) produces the analysis
"""

import os
import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user

router = APIRouter(prefix="/vision", tags=["Vision"])


class VisionResponse(BaseModel):
    extraction: str
    analysis: str
    model: str


# Supported image types
_SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _extract_from_image(image_bytes: bytes) -> str:
    """
    Step 1: Extract structured text from the image using Azure GPT-4o Vision.

    GPT-4o is a strong multimodal model that excels at reading charts, tables,
    financial statements, and any visual data. We use it purely for extraction
    (no interpretation) — the analysis step is handled by Groq LLaMA separately.
    """
    import base64
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "")

    if not api_key or not base_url:
        # Fallback: try the existing OpenRouter VisionAnalyzer
        from OCR_Processing.vision_client import VisionAnalyzer
        vision = VisionAnalyzer()
        extraction_prompt = (
            "You are a precise OCR extraction system. Extract ALL visible content from this image:\n"
            "- Charts/graphs: axes, labels, data points, values, trends, legend\n"
            "- Tables: reproduce in Markdown format\n"
            "- Financial statements: all line items with values\n"
            "- Text: transcribe verbatim\n\n"
            "Output ONLY extracted content. No interpretation."
        )
        return vision.analyze_image(image_bytes, extraction_prompt)

    # Fix the base URL — remove '/responses' suffix if present (Azure AI inference quirk)
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/responses"):
        clean_base = clean_base[: -len("/responses")]

    client = OpenAI(api_key=api_key, base_url=clean_base)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    extraction_prompt = (
        "You are a precise OCR and data extraction system. Extract ALL visible content from this image:\n"
        "- If it's a chart/graph: list axes labels, data points, values, trends, legend entries\n"
        "- If it's a table: reproduce it exactly in Markdown table format\n"
        "- If it's a financial statement: extract every line item with its value\n"
        "- If it contains text: transcribe it verbatim\n\n"
        "Output ONLY the extracted content. No interpretation, no opinions, no analysis. "
        "Be exhaustive — capture every number, label, and data point visible."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=2000,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Vision] Azure GPT-4o extraction failed: {e}")
        # Fallback to OpenRouter VisionAnalyzer
        from OCR_Processing.vision_client import VisionAnalyzer
        vision = VisionAnalyzer()
        return vision.analyze_image(
            image_bytes,
            "Extract ALL text, numbers, labels, and data from this image. Output only raw content.",
        )


def _generate_analysis(extracted_text: str, user_question: str) -> tuple[str, str]:
    """
    Step 2: Pass the extracted content to Groq LLaMA 3.3-70B for financial analysis.
    Same model used by the main RAG pipeline — ensures consistent answer quality.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not configured.",
        )

    from agents.financial_agent import _get_llm, _FALLBACK_MODELS

    system_prompt = (
        "You are an expert financial analyst. You have been given text extracted from a financial image "
        "(chart, graph, table, or report screenshot). Analyze this extracted data and provide:\n\n"
        "- A clear summary of what the data shows\n"
        "- Key metrics, values, and trends identified\n"
        "- Financial interpretation and insights\n"
        "- Notable patterns, anomalies, or signals relevant to investment decisions\n\n"
        "Be precise with numbers. Structure your response with clear sections.\n"
        "Respond in the same language as the user's question (if provided)."
    )

    user_content = f"## Extracted Image Content\n\n{extracted_text}"
    if user_question:
        user_content += f"\n\n## User Question\n\n{user_question}"
    user_content += "\n\n## Instructions\n\nProvide a detailed financial analysis of the above extracted data."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    # Try Groq first (fast), then fallbacks — same logic as the main generate_answer
    openrouter_primary = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    models_to_try = [None] + [m for m in _FALLBACK_MODELS if m != openrouter_primary] + [openrouter_primary]

    for model_name in models_to_try:
        try:
            llm = _get_llm(model_name)
            response = llm.invoke(messages)
            used_model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return response.content, used_model
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg or "429" in error_msg or "502" in error_msg:
                continue
            if model_name == models_to_try[-1]:
                raise HTTPException(status_code=502, detail=f"All LLM models failed: {error_msg[:200]}")
            continue

    raise HTTPException(status_code=502, detail="All LLM models exhausted.")


@router.post("/analyze", response_model=VisionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
) -> VisionResponse:
    """
    Upload a financial image and receive a two-step AI analysis:
      1. OCR/Vision extraction — reads all content from the image
      2. LLM analysis — interprets the extracted data financially

    Supported formats: PNG, JPEG, GIF, WebP
    Max size: 10MB
    """
    import os.path
    from api.tracing import trace_query_span, trace_stage, set_output

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Supported: {', '.join(sorted(_SUPPORTED_IMAGE_TYPES))}",
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    with trace_query_span(
        user=user,
        question=question or f"[Image Analysis] {file.filename}",
        persona="Vision",
        focus_mode="Image Analysis",
        collection="N/A",
    ) as root_span:
        root_span.set_attribute("vision.filename", file.filename or "unknown")
        root_span.set_attribute("vision.file_size_kb", round(len(content) / 1024, 1))
        root_span.set_attribute("vision.file_type", ext)
        if question:
            root_span.set_attribute("vision.user_question", question[:300])

        # ── Step 1: Extract text/data from image ──────────────────────────
        with trace_stage("vision_extraction", {"extraction.model": "gpt-4o"}) as ext_span:
            try:
                extracted_text = _extract_from_image(content)
                ext_span.set_attribute("extraction.output_length", len(extracted_text or ""))
                ext_span.set_attribute("extraction.success", True)
            except Exception as e:
                ext_span.set_attribute("extraction.success", False)
                ext_span.set_attribute("extraction.error", str(e)[:300])
                raise HTTPException(
                    status_code=500,
                    detail=f"Image extraction failed: {str(e)[:200]}",
                )

        if not extracted_text or len(extracted_text.strip()) < 10:
            root_span.set_attribute("vision.result", "no_content")
            raise HTTPException(
                status_code=422,
                detail="Could not extract meaningful content from the image. Try a clearer image.",
            )

        # ── Step 2: Analyze extracted content with Groq LLM ───────────────
        with trace_stage("vision_analysis") as gen_span:
            try:
                analysis, model_used = _generate_analysis(extracted_text, question or "")
                gen_span.set_attribute("analysis.model", model_used)
                gen_span.set_attribute("analysis.output_length", len(analysis))
                gen_span.set_attribute("analysis.success", True)
            except HTTPException:
                gen_span.set_attribute("analysis.success", False)
                raise
            except Exception as e:
                gen_span.set_attribute("analysis.success", False)
                gen_span.set_attribute("analysis.error", str(e)[:300])
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis generation failed: {str(e)[:200]}",
                )

        # Set output on root span
        root_span.set_attribute("vision.extraction_model", "gpt-4o")
        root_span.set_attribute("vision.analysis_model", model_used)
        root_span.set_attribute("vision.result", "success")
        set_output(root_span, analysis[:2000])

    return VisionResponse(
        extraction=extracted_text,
        analysis=analysis,
        model=model_used,
    )
