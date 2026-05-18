"""
api/routes/pdf_viewer.py
Endpoint that serves a specific page of an ingested PDF file for in-UI preview.
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse, Response

from api.auth import get_current_user

router = APIRouter()

# Root directory where uploaded/collected PDFs live
DATA_ROOT = Path(__file__).parent.parent.parent  # project root


@router.get("/pdf/file")
@router.head("/pdf/file")
def serve_pdf(
    path: str = Query(..., description="Absolute or relative path to the PDF"),
    user: dict = Depends(get_current_user),
):
    """
    Serve a PDF file so the browser can display it inline.
    The frontend will append #page=N to jump directly to the right page.
    """
    # ── Path resolution ────────────────────────────────────────────────────────
    # The path may arrive as:
    #   a) A Windows absolute path: C:\Users\...\data\uploads\file.pdf
    #   b) A relative path:         data/uploads/file.pdf
    #   c) A Linux absolute path:   /app/data/uploads/file.pdf
    #
    # Inside Docker the Windows path won't exist; we always fall back to a
    # recursive search by filename stem inside /app/data.

    pdf_path = Path(path)

    # Detect Windows-style absolute paths (e.g. C:\...) running inside Docker
    is_windows_path = len(path) > 1 and path[1] == ':'

    if not is_windows_path and not pdf_path.is_absolute():
        pdf_path = DATA_ROOT / pdf_path

    # Security: only enforce for non-Windows paths that are actually resolvable
    if not is_windows_path and pdf_path.is_absolute():
        try:
            pdf_path.resolve().relative_to(DATA_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path outside project")

    if not pdf_path.exists() or not pdf_path.is_file():
        # Fallback: Deep recursive search inside the /data directory by stem
        stem = Path(path).stem  # extract just the filename without extension
        data_dir = DATA_ROOT / "data"
        if data_dir.exists():
            candidates = list(data_dir.rglob(f"*{stem[:30]}*.pdf"))
            if candidates:
                pdf_path = candidates[0]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF not found on disk: {Path(path).name}. The source document may have been deleted or moved."
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"PDF not found on disk: {Path(path).name}."
            )

    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={pdf_path.name}"}
    )
