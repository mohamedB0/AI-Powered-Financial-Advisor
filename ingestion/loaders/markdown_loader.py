"""
ingestion/loaders/markdown_loader.py
Loads and parses Markdown files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """
    Loads Markdown files and returns (text, metadata) tuples.
    
    Preserves heading structure as metadata.
    """

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Load a Markdown file and return sections.
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        results: List[Tuple[str, Dict[str, Any]]] = []
        
        # Split by headings
        sections = re.split(r"(?=^#{1,6}\s)", content, flags=re.MULTILINE)
        
        current_heading = "Untitled"
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Extract heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", section, re.MULTILINE)
            if heading_match:
                current_heading = heading_match.group(2).strip()
                section = re.sub(r"^#{1,6}\s+.+$\n?", "", section, flags=re.MULTILINE).strip()
            
            if section:
                metadata = {
                    "source": path.name,
                    "doc_type": "markdown",
                    "heading": current_heading,
                    "file_path": str(path),
                }
                results.append((section, metadata))
        
        logger.info("MarkdownLoader: loaded %d sections from '%s'", len(results), path.name)
        return results
