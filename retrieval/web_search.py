"""
retrieval/web_search.py
Web search using DuckDuckGo HTML (no API key required).
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional


class WebSearch:
    """
    Simple web search using DuckDuckGo HTML.
    Falls back gracefully if search fails.
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web and return results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dicts with 'title', 'url', 'snippet'
        """
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.select(".result")[:max_results]:
                title_elem = result.select_one(".result__title a")
                snippet_elem = result.select_one(".result__snippet")
                
                if title_elem:
                    results.append({
                        "title": self._clean_text(title_elem.get_text()),
                        "url": title_elem.get("href", ""),
                        "snippet": self._clean_text(snippet_elem.get_text()) if snippet_elem else "",
                    })
            
            return results
        except Exception as e:
            print(f"[WebSearch] Search failed: {e}")
            return []
    
    def search_financial_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for financial news specifically.
        """
        financial_query = f"{query} stock market finance"
        return self.search(financial_query, max_results)
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as readable text."""
        if not results:
            return "No web search results found."
        
        lines = ["## Web Search Results\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**{i}. {r['title']}**")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   Source: {r['url']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_morocco_market_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Scrape actual Casablanca Stock Exchange data for Morocco market queries.
        Returns real stock data (gainers, losers, most active) instead of search links.
        """
        try:
            import warnings
            from urllib3.exceptions import InsecureRequestWarning
            warnings.filterwarnings('ignore')
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }

            q_lower = query.lower()
            # Maps user keywords to BVC API ticker codes + name keywords for fallback matching
            bvc_tickers = {
                "ATW": {"pattern": r"\b(atw|attijariwafa|attijari)\b", "names": ["attijariwafa", "attijari"]},
                "BCP": {"pattern": r"\b(bcp|bcm|banque\.?populaire|cm\.?cp|banque\.?centrale\.?populaire)\b", "names": ["populaire", "bcp"]},
                "CFG": {"pattern": r"\b(cfg|cfagr?oup|credit\.?financier|cfg\.?bank)\b", "names": ["cfg", "credit du maroc"]},
                "CIH": {"pattern": r"\b(cih|cih\.?bank)\b", "names": ["cih"]},
                "IAM": {"pattern": r"\b(iam|ita|itissalat|maroc\.?telecom)\b", "names": ["itissalat", "maroc telecom"]},
                "LHM": {"pattern": r"\b(lhm|lafarge|lafarge\.?holcim\.?maroc)\b", "names": ["lafarge", "holcim"]},
                "MNG": {"pattern": r"\b(mng|managem)\b", "names": ["managem"]},
                "SAM": {"pattern": r"\b(sam|samir|raffinerie)\b", "names": ["samir"]},
                "SLF": {"pattern": r"\b(slf|salafin)\b", "names": ["salafin"]},
                "WAA": {"pattern": r"\b(waa|wav|wafasalaf|wafa\.?assurance)\b", "names": ["wafa assurance", "wafa"]},
                "TQM": {"pattern": r"\b(tqm|taqa\.?morocco|taqa)\b", "names": ["taqa"]},
                "BOA": {"pattern": r"\b(boa|bank\.?of\.?africa|bmce)\b", "names": ["bank of africa", "bmce"]},
                "MSA": {"pattern": r"\b(msa|marsa\.?maroc)\b", "names": ["marsa maroc"]},
                "ADH": {"pattern": r"\b(adh|addoha)\b", "names": ["addoha"]},
                "LBV": {"pattern": r"\b(lbv|label\.?vie)\b", "names": ["label'vie", "label vie"]},
            }
            single_ticker = None
            single_names = []
            for ticker, info in bvc_tickers.items():
                if re.search(info["pattern"], q_lower):
                    single_ticker = ticker
                    single_names = info["names"]
                    break

            if single_ticker:
                endpoint = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker"
                params = {"marche": 59, "class[]": 50}
                label = f"{single_ticker} - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["loser", "down", "fall", "declin", "baisse", "baissier", "perte", "chute", "recul"]):
                endpoint = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker"
                params = {"marche": 59, "class[]": 50}
                label = "Top Losers - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["gainer", "up", "rise", "gain", "top mover", "mover", "hausse", "haussier", "monte", "progression", "meilleur"]):
                endpoint = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker"
                params = {"marche": 59, "class[]": 50}
                label = "Top Gainers - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["active", "volume", "traded", "actif", "échangé", "echange"]):
                endpoint = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker"
                params = {"marche": 59, "class[]": 50}
                label = "Most Active Stocks - Casablanca Stock Exchange"
            else:
                endpoint = "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker"
                params = {"marche": 59, "class[]": 50}
                label = "Casablanca Stock Exchange - Market Summary"

            response = requests.get(endpoint, params=params, headers=headers, verify=False, timeout=15)
            if response.status_code != 200:
                return []

            stocks = response.json().get("data", {}).get("values", [])
            if not stocks:
                return []

            for s in stocks:
                s["last_price"] = s.get("field_cours_courant")
                s["variation_pct"] = s.get("field_var_veille")
                s["volume"] = s.get("field_cumul_volume_echange")

            if single_ticker:
                # Try exact ticker match first
                filtered = [s for s in stocks if s.get("ticker", "").upper() == single_ticker]
                # Fallback: match by company name keywords
                if not filtered and single_names:
                    filtered = [
                        s for s in stocks
                        if any(kw in (s.get("label", "") + s.get("name", "")).lower() for kw in single_names)
                    ]
                stocks = filtered[:5]
                label = f"{single_ticker} Stock - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["loser", "down", "fall", "declin", "baisse", "baissier", "perte", "chute", "recul"]):
                stocks = [s for s in stocks if s.get("variation_pct") is not None and float(s["variation_pct"]) < 0]
                stocks.sort(key=lambda x: float(x["variation_pct"]))
                label = "Top Losers - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["gainer", "up", "rise", "gain", "top mover", "mover", "hausse", "haussier", "monte", "progression", "meilleur"]):
                stocks = [s for s in stocks if s.get("variation_pct") is not None and float(s["variation_pct"]) > 0]
                stocks.sort(key=lambda x: float(x["variation_pct"]), reverse=True)
                label = "Top Gainers - Casablanca Stock Exchange"
            elif any(r in q_lower for r in ["active", "volume", "traded", "actif", "échangé", "echange"]):
                stocks = [s for s in stocks if s.get("volume") is not None]
                stocks.sort(key=lambda x: float(x["volume"] or 0), reverse=True)
                label = "Most Active Stocks - Casablanca Stock Exchange"

            stocks = stocks[:10]

            formatted = []
            for s in stocks:
                ticker = s.get("ticker", "N/A")
                name = s.get("label", s.get("name", "N/A"))
                price = s.get("last_price", "N/A")
                change = s.get("variation_pct", "N/A")
                volume = s.get("volume", "N/A")
                change_str = f"{float(change):+.2f}%" if change not in [None, "N/A"] else "N/A"
                formatted.append({
                    "title": label,
                    "url": "https://www.casablanca-bourse.com",
                    "snippet": f"{ticker} | {name} | Price: {price} | Change: {change_str} | Volume: {volume}"
                })

            return formatted

        except Exception as e:
            print(f"[WebSearch] Morocco market scrape failed: {e}")
            return []
