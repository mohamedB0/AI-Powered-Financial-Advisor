"""
api/routes/pdf_viewer.py
Endpoint that serves a specific page of an ingested PDF file for in-UI preview.
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

router = APIRouter()

# Root directory where uploaded/collected PDFs live
DATA_ROOT = Path(__file__).parent.parent.parent  # project root


@router.get("/pdf/file")
@router.head("/pdf/file")
def serve_pdf(
    path: str = Query(..., description="Filename or relative path to the PDF within the data directory"),
):
    """
    Serve a PDF file so the browser can display it inline.
    The frontend will append #page=N to jump directly to the right page.

    SECURITY: Only serves PDFs within the project's data directory.
    Absolute paths and path traversal sequences are rejected.
    """
    # ── Security: reject absolute paths and traversal attempts ─────────────────
    # Strip any path traversal sequences
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise HTTPException(status_code=403, detail="Access denied: invalid path")

    # Reject Windows absolute paths (e.g. C:\...)
    if len(path) > 1 and path[1] == ':':
        # Extract just the filename for safety — ignore the absolute path
        path = Path(path).name

    # Sanitize: only allow the filename or a relative subpath within data/
    safe_name = Path(path).name  # Extract filename only, strip any directory components

    # ── Search for the file within allowed directories only ────────────────────
    ALLOWED_DIRS = [
        DATA_ROOT / "data",
        DATA_ROOT / "data" / "uploads",
        DATA_ROOT / "data" / "collected_data",
    ]

    pdf_path = None

    # First try: direct match by relative path within DATA_ROOT/data
    candidate = DATA_ROOT / "data" / path
    if candidate.exists() and candidate.is_file():
        try:
            candidate.resolve().relative_to((DATA_ROOT / "data").resolve())
            pdf_path = candidate
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    # Second try: search by filename stem within allowed directories
    if pdf_path is None:
        stem = Path(safe_name).stem[:30]  # Limit stem length
        for allowed_dir in ALLOWED_DIRS:
            if allowed_dir.exists():
                candidates = list(allowed_dir.rglob(f"{stem}*.pdf"))
                if candidates:
                    pdf_path = candidates[0]
                    break

    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found: {safe_name}. The source document may have been deleted or moved."
        )

    # Final security check: ensure resolved path is within data directory
    try:
        pdf_path.resolve().relative_to((DATA_ROOT / "data").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={pdf_path.name}"}
    )
