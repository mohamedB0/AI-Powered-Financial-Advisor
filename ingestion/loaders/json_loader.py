"""
ingestion/loaders/json_loader.py
Loads and parses JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class JSONLoader:
    """
    Loads JSON files and returns text representations.
    
    Handles:
    - Simple key-value objects
    - Arrays of objects
    - Nested structures
    """

    def __init__(self, flatten: bool = True):
        self.flatten = flatten

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Load a JSON file and return text chunks.
        Stores structured data in PostgreSQL.
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("JSONLoader: Invalid JSON in '%s': %s", path.name, e)
            return []

        results: List[Tuple[str, Dict[str, Any]]] = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                text = self._object_to_text(item)
                metadata = {
                    "source": path.name,
                    "doc_type": "json",
                    "file_path": str(path),
                    "index": i,
                }
                results.append((text, metadata))
        elif isinstance(data, dict):
            text = self._object_to_text(data)
            metadata = {
                "source": path.name,
                "doc_type": "json",
                "file_path": str(path),
            }
            results.append((text, metadata))
        else:
            text = str(data)
            metadata = {
                "source": path.name,
                "doc_type": "json",
                "file_path": str(path),
            }
            results.append((text, metadata))

        # Store in PostgreSQL
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()
            
            doc_id = db.insert_document(
                filename=path.name,
                file_type="json",
                file_size=path.stat().st_size,
                metadata={"record_count": len(results)},
            )
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    db.insert_json_data(
                        document_id=doc_id,
                        source_file=path.name,
                        data=item,
                        key_path=f"[{i}]",
                    )
            else:
                db.insert_json_data(
                    document_id=doc_id,
                    source_file=path.name,
                    data=data,
                    key_path="root",
                )
            
            db.update_document_status(doc_id, "completed", chunks_count=len(results))
            logger.info("JSONLoader: Stored %d records in PostgreSQL", len(results))
            
        except Exception as e:
            logger.warning("JSONLoader: Failed to store in SQL: %s", e)

        logger.info("JSONLoader: loaded %d records from '%s'", len(results), path.name)
        return results

    def _object_to_text(self, obj: Any, prefix: str = "") -> str:
        """
        Convert a JSON object to readable text.
        """
        lines = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._object_to_text(value, prefix + "  "))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                lines.append(f"{prefix}[{i}]: {self._object_to_text(item, prefix + '  ')}")
        else:
            lines.append(f"{prefix}{obj}")
        
        return "\n".join(lines)
