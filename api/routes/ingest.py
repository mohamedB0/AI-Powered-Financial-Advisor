"""
api/routes/ingest.py
POST /ingest — receives an uploaded file and runs the ingestion pipeline.

Supports two modes:
1. Sync: Runs in thread-pool (current behavior)
2. Async: Offloads to Celery worker for background processing

Set USE_CELERY=true in .env to enable async mode.
"""

import os
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

USE_CELERY = os.getenv("USE_CELERY", "false").lower() == "true"


class IngestResponse(BaseModel):
    filename: str
    chunks: int
    vectors: int
    message: str


class AsyncIngestResponse(BaseModel):
    task_id: str
    status: str
    message: str
    status_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    result: Optional[dict] = None
    info: Optional[str] = None
    step: Optional[str] = None
    detail: Optional[str] = None
    percent: Optional[int] = None


def _run_pipeline(tmp_path: str, ext: str) -> tuple[int, int]:
    """Blocking ingestion pipeline — runs in a thread-pool."""
    from ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    result = pipeline.ingest_file(tmp_path)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except:
        pass
    
    return result["chunks"], result["stored"]


@router.post("/", response_model=IngestResponse)
async def ingest_file_sync(file: UploadFile = File()) -> IngestResponse:
    """
    Upload a document and ingest it into the vector store.
    
    Supported formats: PDF, HTML, CSV, Excel, Word, Markdown, JSON, Images
    
    This is the synchronous endpoint - the API will wait for ingestion to complete.
    For large files, consider using /ingest/async instead.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    supported = {
        ".pdf", ".html", ".htm",  # Documents
        ".csv", ".tsv", ".xlsx", ".xls",  # Spreadsheets
        ".docx",  # Word documents
        ".md", ".markdown",  # Markdown
        ".json",  # JSON
        ".png", ".jpg", ".jpeg", ".tiff",  # Images (OCR)
    }
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(supported))}",
        )

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        loop = asyncio.get_event_loop()
        num_chunks, num_vectors = await loop.run_in_executor(
            None, _run_pipeline, tmp_path, ext
        )

        return IngestResponse(
            filename=file.filename or "",
            chunks=num_chunks,
            vectors=num_vectors,
            message="Document ingested successfully.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async", response_model=AsyncIngestResponse)
async def ingest_file_async(
    file: UploadFile = File(),
    chunker_type: str = Query("recursive", enum=["recursive", "semantic"]),
    chunk_size: int = Query(512, ge=64, le=4096),
    chunk_overlap: int = Query(64, ge=0, le=512),
) -> AsyncIngestResponse:
    """
    Upload a file for asynchronous ingestion via Celery.
    
    Supported formats: PDF, HTML, CSV, Excel, Word, Markdown, JSON, Images
    
    This endpoint returns immediately with a task_id. Use /ingest/status/{task_id}
    to check the status of the ingestion.
    """
    if not USE_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Celery is not enabled. Set USE_CELERY=true in .env and restart the API.",
        )
    
    ext = os.path.splitext(file.filename or "")[1].lower()
    supported = {
        ".pdf", ".html", ".htm",
        ".csv", ".tsv", ".xlsx", ".xls",
        ".docx", ".md", ".markdown", ".json",
        ".png", ".jpg", ".jpeg", ".tiff",
    }
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(supported))}",
        )

    try:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        from tasks.document_ingestion_tasks import ingest_document
        
        task = ingest_document.apply_async(
            args=[str(file_path)],
        )
        
        return AsyncIngestResponse(
            task_id=task.id,
            status="PENDING",
            message="Ingestion task submitted. Check status_url for progress.",
            status_url=f"/ingest/status/{task.id}",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the status of an async ingestion task.
    
    Args:
        task_id: The Celery task ID returned from /ingest/async
    
    Returns:
        Task status and result (if completed)
    """
    if not USE_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Celery is not enabled. Set USE_CELERY=true in .env and restart the API.",
        )
    
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id)
    
    # Extract rich progress info if task is in PROGRESS state
    step = None
    detail = None
    percent = None
    if result.state == "PROGRESS" and isinstance(result.info, dict):
        step = result.info.get("step")
        detail = result.info.get("detail")
        percent = result.info.get("percent")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=result.state,
        ready=result.ready(),
        result=result.result if result.ready() and isinstance(result.result, dict) else None,
        info=str(result.info) if result.info and result.state not in ("PROGRESS",) else None,
        step=step,
        detail=detail,
        percent=percent,
    )


@router.post("/batch", response_model=AsyncIngestResponse)
async def ingest_batch_async(
    files: list[UploadFile] = File(...),
    chunker_type: str = Query("recursive", enum=["recursive", "semantic"]),
) -> AsyncIngestResponse:
    """
    Upload multiple files for batch async ingestion.
    
    All files are submitted as a single Celery task.
    """
    if not USE_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Celery is not enabled. Set USE_CELERY=true in .env and restart the API.",
        )
    
    try:
        upload_dir = Path("data/uploads/batch")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_paths = []
        
        for i, file in enumerate(files):
            safe_filename = f"{timestamp}_{i}_{file.filename}"
            file_path = upload_dir / safe_filename
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(str(file_path))
        
        from tasks.document_ingestion_tasks import ingest_document
        
        # Submit each file as a separate task
        task_ids = []
        for fp in file_paths:
            t = ingest_document.apply_async(args=[fp])
            task_ids.append(t.id)
        
        # Return the last task id for status tracking
        task = type('obj', (object,), {'id': task_ids[-1]})()
        
        return AsyncIngestResponse(
            task_id=task.id,
            status="PENDING",
            message=f"Batch ingestion of {len(files)} files submitted.",
            status_url=f"/ingest/status/{task.id}",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/stats")
async def get_queue_stats():
    """
    Get Celery queue statistics.
    
    Requires Flower to be running or direct Redis access.
    """
    if not USE_CELERY:
        raise HTTPException(
            status_code=503,
            detail="Celery is not enabled.",
        )
    
    try:
        from tasks.celery_app import celery_app
        
        inspect = celery_app.control.inspect()
        
        return {
            "active_workers": inspect.active() or {},
            "active_workers_count": len(inspect.active() or {}),
            "queued_tasks": inspect.reserved() or {},
            "registered_tasks": inspect.registered() or {},
        }
    except Exception as e:
        return {"error": str(e)}
