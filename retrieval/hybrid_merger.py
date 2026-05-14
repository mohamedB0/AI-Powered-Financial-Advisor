"""
retrieval/hybrid_merger.py
Merges vector search, SQL, and web search results into a single ranked context string.
"""

from typing import List, Dict, Any, Optional


class HybridMerger:
    """
    Combines and de-duplicates results from vector search, SQL, and web search.
    Vector results are ranked by semantic score; SQL data is appended as
    structured context; web results are included for current information.
    """

    def merge(
        self,
        vector_results: List[Dict[str, Any]],
        sql_results: List[Dict[str, Any]],
        web_results: Optional[List[Dict[str, Any]]] = None,
        max_vector: int = 4,
        max_sql: int = 3,
        max_web: int = 3,
    ) -> str:
        """
        Build a merged context string from the retrieval streams that have data.
        Only displays sections with actual results.

        Args:
            vector_results: Results from VectorSearch.search()
            sql_results:    Results from SQLSearch query methods
            web_results:    Results from WebSearch
            max_vector:     Max vector chunks to include
            max_sql:        Max SQL rows to include
            max_web:        Max web results to include

        Returns:
            Formatted context string ready to pass to an LLM.
        """
        parts = []
        web_results = web_results or []
        
        # Track which sources have data
        has_vector = bool(vector_results)
        has_sql = bool(sql_results)
        has_web = bool(web_results)
        
        # If nothing was retrieved, return a generic message
        if not (has_vector or has_sql or has_web):
            return "No relevant data found. Please try rephrasing your question."

        # ── Vector (Document) Context ─────────────────────────────────────────
        if has_vector:
            parts.append("### 📄 Retrieved Documents Summary\n")
            parts.append("| # | Source | Metadata | Score |")
            parts.append("|---|--------|----------|-------|")
            
            seen_texts = set()
            filtered_results = []
            
            for r in vector_results:
                text = r.get("text", "").strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                filtered_results.append(r)
                if len(filtered_results) >= max_vector:
                    break
                    
            for count, r in enumerate(filtered_results, 1):
                # Safely get metadata, fallback to empty dict if it's purely a string or missing
                meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else r
                
                # Fetch various possible keys across old and new pipelines
                source = meta.get("source_file") or meta.get("source") or "unknown"
                page = meta.get("page", "")
                
                # Build extra metadata tags
                tags = []
                if page: tags.append(f"Pg {page}")
                if meta.get("ocr") == "true" or meta.get("extraction_method") == "ocr_vision": tags.append("OCR/Vision")
                if meta.get("year"): tags.append(f"{meta.get('year')}")
                
                meta_str = ", ".join(tags) if tags else "N/A"
                
                # Use rerank_score if available (from Jina), otherwise base vector score
                raw_score = meta.get("rerank_score") or r.get("score", 0)
                score_str = f"{float(raw_score):.4f}"
                parts.append(f"| {count} | {source} | {meta_str} | {score_str} |")
                
            parts.append("\n### 🔍 Document Excerpts\n")
            for count, r in enumerate(filtered_results, 1):
                meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else r
                source = meta.get("source_file") or meta.get("source") or "unknown"

                # Only show the concise fields the user cares about
                allowed_keys = ("page", "collection", "rerank_score", "doc_type")
                meta_details = []
                for k in allowed_keys:
                    v = meta.get(k)
                    if v is not None and v != "":
                        if k == "rerank_score":
                            meta_details.append(f"score: {float(v):.4f}")
                        else:
                            meta_details.append(f"{k}: {v}")

                meta_label = (" | " + " | ".join(meta_details)) if meta_details else ""

                text = r.get("text", "").strip()
                quoted_text = text.replace('\n', '\n> ')

                # Build a clickable link if we have a file path OR at least a source filename
                file_path = meta.get("file_path") or meta.get("path") or source or ""
                page_num  = meta.get("page", 1) or 1
                if file_path and file_path != "unknown":
                    import urllib.parse
                    encoded_path = urllib.parse.quote(str(file_path), safe="")
                    pdf_link = f"[📄 {source}{meta_label}](pdf-viewer:{encoded_path}&page={page_num})"
                    parts.append(f"**Excerpt {count}** — {pdf_link}")
                else:
                    parts.append(f"**Excerpt {count}** — `[SOURCE: {source}{meta_label}]`")
                parts.append(f"> {quoted_text}\n")

        # ── SQL (Structured Data) Context ─────────────────────────────────────
        if has_sql:
            parts.append("## Retrieved Data (PostgreSQL)\n")
            
            # Detect which table the data came from based on columns
            first_row = sql_results[0]
            tables_used = []
            
            if any(k in first_row for k in ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']):
                tables_used.append("stock_prices")
            if any(k in first_row for k in ['revenue', 'ebitda', 'net_income', 'eps_diluted', 'company']):
                tables_used.append("financial_metrics")
            if any(k in first_row for k in ['country_name', 'indicator_name', 'value']):
                tables_used.append("macro_indicators")
            if any(k in first_row for k in ['sheet_name', 'row_index', 'data_json']):
                tables_used.append("excel_data")
            
            if tables_used:
                parts.append(f"**Tables used:** {', '.join(tables_used)}\n")
            
            headers = list(first_row.keys())
            header_row = " | ".join(f"**{h}**" for h in headers)
            separator = " | ".join("---" for _ in headers)
            parts.append(header_row)
            parts.append(separator)
            for row in sql_results[:max_sql]:
                parts.append(" | ".join(str(row.get(h, "")) for h in headers))
            parts.append("")

        # ── Web Search Context ────────────────────────────────────────────────
        if has_web:
            parts.append("## Web Search Results\n")
            for i, r in enumerate(web_results[:max_web], 1):
                parts.append(f"**{i}. {r.get('title', 'No title')}**")
                parts.append(f"   {r.get('snippet', '')}")
                parts.append(f"   Source: {r.get('url', '')}")
                parts.append("")

        return "\n".join(parts)
