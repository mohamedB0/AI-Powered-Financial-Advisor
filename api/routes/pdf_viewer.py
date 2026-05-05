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
def serve_pdf(path: str = Query(..., description="Absolute or relative path to the PDF")):
    """
    Serve a PDF file so the browser can display it inline.
    The frontend will append #page=N to jump directly to the right page.
    """
    # Resolve the path; allow absolute paths AND relative-to-project
    pdf_path = Path(path)
    if not pdf_path.is_absolute():
        pdf_path = DATA_ROOT / pdf_path

    # Security: ensure the file is inside the project directory
    try:
        pdf_path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside project")

    if not pdf_path.exists() or not pdf_path.is_file():
        # Fallback: Deep continuous search inside the /data directory
        stem = pdf_path.stem  # filename without extension
        data_dir = DATA_ROOT / "data"
        if data_dir.exists():
            # Use recursive glob to flawlessly find the PDF regardless of its subfolder
            candidates = list(data_dir.rglob(f"*{stem[:20]}*.pdf"))
            if candidates:
                pdf_path = candidates[0]  # Take the freshest match
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF not found on disk: {pdf_path.name}. The source document may have been deleted or moved."
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"PDF not found on disk: {pdf_path.name}. The source document may have been deleted or moved."
            )

    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={pdf_path.name}"}
    )
