"""
agents/tools.py
Tool definitions used by the LangGraph financial analysis agent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Annotated
from langchain_core.tools import tool
from retrieval.sql_search import SQLSearch
from retrieval.web_search import WebSearch

_sql = None
_web = None

def _get_sql():
    global _sql
    if _sql is None:
        _sql = SQLSearch()
    return _sql

def _get_web():
    global _web
    if _web is None:
        _web = WebSearch()
    return _web


@tool
def get_stock_price(ticker: Annotated[str, "Stock ticker symbol (e.g. AAPL, MSFT)"]) -> str:
    """Get the latest closing price and recent history for a stock ticker."""
    latest = _get_sql().get_latest_price(ticker)
    history = _get_sql().get_price_history(ticker, days=5)
    if not latest:
        return f"No price data found for {ticker}."
    result = f"**{ticker} — Latest Price**\n"
    result += _get_sql().format_results(latest) + "\n\n"
    result += f"**{ticker} — Last 5 Days**\n"
    result += _get_sql().format_results(history)
    return result


@tool
def get_company_financials(
    company: Annotated[str, "Company name (e.g. Apple, Amazon)"]
) -> str:
    """Get annual financial metrics (revenue, EBITDA, net income) for a company."""
    rows = _get_sql().get_financial_metrics(company)
    if not rows:
        return f"No financial metrics found for '{company}'."
    return f"**Financial Metrics — {company}**\n" + _get_sql().format_results(rows)


@tool
def get_top_companies_by_revenue(
    year: Annotated[int, "Fiscal year (e.g. 2023)"] = 2023
) -> str:
    """Get the top companies ranked by annual revenue for a given year."""
    rows = _get_sql().get_top_by_revenue(year)
    if not rows:
        return f"No revenue data found for {year}."
    return f"**Top Companies by Revenue ({year})**\n" + _get_sql().format_results(rows)


@tool
def get_market_movers() -> str:
    """Get the top performing stocks by percentage change over the last 30 days."""
    rows = _get_sql().get_top_movers()
    if not rows:
        return "No market mover data available."
    return "**Top Market Movers (Last 30 Days)**\n" + _get_sql().format_results(rows)


@tool
def get_uploaded_documents() -> str:
    """Get a list of all uploaded documents and their metadata."""
    docs = _get_sql().get_document_list()
    if not docs:
        return "No documents uploaded yet."
    result = "**Uploaded Documents**\n\n"
    result += "| Filename | Type | Status | Rows |\n"
    result += "|---------|------|--------|------|\n"
    for doc in docs:
        result += f"| {doc['filename']} | {doc['file_type']} | {doc['status']} | {doc.get('chunks_count', 0)} |\n"
    return result


@tool
def query_document_data(
    filename: Annotated[str, "Part of the filename to search (e.g. 'dette', 'monnaie', 'comptes')"]
) -> str:
    """Query data from uploaded Excel/CSV documents. Use this when user asks about specific uploaded data."""
    docs = _get_sql().get_document_list()
    matching_docs = [d for d in docs if filename.lower() in d['filename'].lower()]
    if not matching_docs:
        return f"No document found matching '{filename}'."
    doc = matching_docs[0]
    data = _get_sql().get_excel_data(doc['id'])
    if not data:
        return f"No data found in {doc['filename']}."
    result = f"**Data from: {doc['filename']}**\n"
    result += f"(Total rows: {len(data)})\n\n"
    for i, row in enumerate(data[:5]):
        result += f"\nRow {i+1}: {row['data_json']}\n"
    if len(data) > 5:
        result += f"\n... and {len(data) - 5} more rows"
    return result


@tool
def web_search(query: Annotated[str, "Search query for current news or information"]) -> str:
    """Search the web for current news, events, or information not available in databases."""
    results = _get_web().search_financial_news(query, max_results=5)
    if not results:
        return f"No web results found for: {query}"
    return _get_web().format_results(results)


TOOLS = [
    get_stock_price,
    get_company_financials,
    get_top_companies_by_revenue,
    get_market_movers,
    get_uploaded_documents,
    query_document_data,
    web_search,
]
