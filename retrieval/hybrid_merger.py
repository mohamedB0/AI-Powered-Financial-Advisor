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
            parts.append("| # | Source | Page | Relevance |")
            parts.append("|---|--------|------|-----------|")
            
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
                source = r.get("source", "unknown")
                page = r.get("page", "N/A")
                score = r.get("score", 0)
                relevance_pct = min(int(score * 100), 100)
                relevance_bar = "█" * (relevance_pct // 10) + "░" * (10 - relevance_pct // 10)
                parts.append(f"| {count} | {source} | {page} | {relevance_bar} {relevance_pct}% |")
                
            parts.append("\n### 🔍 Document Excerpts\n")
            for count, r in enumerate(filtered_results, 1):
                source = r.get("source", "unknown")
                page = r.get("page", "N/A")
                text = r.get("text", "").strip()
                # Prominent source label so LLM can cite it directly
                quoted_text = text.replace('\n', '\n> ')
                parts.append(f"**Excerpt {count}** — `[SOURCE: {source} | Page {page}]`")
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
