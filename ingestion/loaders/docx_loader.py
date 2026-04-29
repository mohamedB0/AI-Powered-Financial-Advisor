"""
ingestion/loaders/docx_loader.py
Extracts text from Word (.docx) files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from docx import Document
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False


class DocxLoader:
    """
    Loads Word documents and returns (text, metadata) tuples.
    
    Extracts:
    - Paragraphs (with heading detection)
    - Tables
    - Document properties
    """

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Load a Word document and return text chunks.
        Stores structured data in PostgreSQL.
        """
        if not _HAS_DOCX:
            raise ImportError("python-docx required: pip install python-docx")

        path = Path(file_path)
        doc = Document(file_path)
        
        results: List[Tuple[str, Dict[str, Any]]] = []
        current_section: List[str] = []
        current_heading = "Untitled"
        
        def save_section():
            nonlocal current_section, current_heading
            if current_section:
                text = "\n".join(current_section)
                meta = {
                    "source": path.name,
                    "doc_type": "docx",
                    "heading": current_heading,
                    "file_path": str(path),
                }
                results.append((text, meta))
                current_section = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            if para.style.name.startswith("Heading"):
                save_section()
                current_heading = text[:100]
            else:
                current_section.append(text)
        
        save_section()
        
        if doc.tables:
            table_texts = []
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(" | ".join(cells))
                table_texts.append("\n".join(rows))
            
            table_combined = "\n\n[TABLES]\n" + "\n\n".join(table_texts)
            meta = {
                "source": path.name,
                "doc_type": "docx",
                "heading": "Tables",
                "file_path": str(path),
            }
            results.append((table_combined, meta))
        
        # Store in PostgreSQL
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()
            
            doc_id = db.insert_document(
                filename=path.name,
                file_type="docx",
                file_size=path.stat().st_size,
                metadata={"sections_count": len(results)},
            )
            
            for text, meta in results:
                db.insert_word_content(
                    document_id=doc_id,
                    source_file=path.name,
                    heading=meta.get("heading", "Untitled"),
                    content=text,
                )
            
            db.update_document_status(doc_id, "completed", chunks_count=len(results))
            logger.info("DocxLoader: Stored %d sections in PostgreSQL", len(results))
            
        except Exception as e:
            logger.warning("DocxLoader: Failed to store in SQL: %s", e)

        logger.info("DocxLoader: loaded %d sections from '%s'", len(results), path.name)
        return results
