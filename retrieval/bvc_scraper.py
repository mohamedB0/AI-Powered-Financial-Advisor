"""
retrieval/bvc_scraper.py
Fetch live market data from Casablanca Bourse (https://www.casablanca-bourse.com).

Data source: Drupal settings JSON embedded in the Equities and Indices pages.
This provides real-time data for all 83 listed stocks including:
  price, open, high, low, variation, volume, quantity, nb trades,
  capitalisation, bid/ask prices, sector, compartment, status.

Cache: 30-second TTL to avoid hammering the BVC website.
"""

import requests
import json
import time
import warnings
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
}

# ── Caches ────────────────────────────────────────────────────────────────
_equities_cache: Dict[str, Any] = {"data": [], "ts": 0}
_indices_cache: Dict[str, Any] = {"data": [], "ts": 0}
_overview_cache: Dict[str, Any] = {"data": None, "ts": 0}
CACHE_TTL = 30  # seconds


# ── Drupal JSON extraction ────────────────────────────────────────────────
def _extract_drupal_settings(url: str) -> Optional[Dict[str, Any]]:
    """Fetch a BVC page and extract the live_market data from Drupal settings JSON."""
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        script = soup.find('script', attrs={'data-drupal-selector': 'drupal-settings-json'})
        if not script or not script.string:
            return None
        settings = json.loads(script.string)
        return settings.get('live_market')
    except Exception as e:
        print(f"BVC Drupal extraction error ({url}): {e}")
        return None


# ── Equities (full stock data) ────────────────────────────────────────────
def _fetch_equities() -> List[Dict[str, Any]]:
    """Fetch all equities from /en/live-market/actions Drupal JSON."""
    live = _extract_drupal_settings(
        'https://www.casablanca-bourse.com/en/live-market/actions'
    )
    if not live or 'actions' not in live:
        return []

    stocks = []
    for item in live['actions']:
        try:
            # Extract multilingual fields (prefer EN, fallback FR)
            name_data = item.get('emetteur', {})
            name = (name_data.get('en') or name_data.get('fr') or '') if isinstance(name_data, dict) else str(name_data)

            sector_data = item.get('secteur', {})
            sector = (sector_data.get('en') or sector_data.get('fr') or '') if isinstance(sector_data, dict) else str(sector_data)

            compartment_data = item.get('compartiment', {})
            compartment = (compartment_data.get('en') or compartment_data.get('fr') or '') if isinstance(compartment_data, dict) else str(compartment_data)

            status_data = item.get('statut', {})
            status = (status_data.get('en') or status_data.get('fr') or 'T') if isinstance(status_data, dict) else str(status_data)

            bid = item.get('achat', {})
            ask = item.get('vente', {})

            stock = {
                "ticker":         item.get('symbol', ''),
                "name":           name,
                "sector":         sector,
                "compartment":    compartment,
                "last_price":     _float(item.get('dernierCours', 0)),
                "ref_price":      _float(item.get('reference', 0)),
                "open":           _float(item.get('ouverture', 0)),
                "high":           _float(item.get('plusHaut', 0)),
                "low":            _float(item.get('plusBas', 0)),
                "variation_pct":  round(_float(item.get('variation', 0)), 4),
                "volume":         _float(item.get('volume', 0)),
                "qty_traded":     _int(item.get('quantite', 0)),
                "nb_trades":      _int(item.get('nbTransactions', 0)),
                "capitalisation": _float(item.get('capitalisation', 0)),
                "nb_shares":      _int(item.get('nombreTitres', 0)),
                "bid_price":      _float(bid.get('prix', 0)) if isinstance(bid, dict) else 0,
                "bid_qty":        _int(bid.get('quantite', 0)) if isinstance(bid, dict) else 0,
                "ask_price":      _float(ask.get('prix', 0)) if isinstance(ask, dict) else 0,
                "ask_qty":        _int(ask.get('quantite', 0)) if isinstance(ask, dict) else 0,
                "status":         status,
            }
            stocks.append(stock)
        except Exception as e:
            print(f"BVC stock parse error ({item.get('symbol', '?')}): {e}")
            continue

    return stocks


def _float(v) -> float:
    """Safe float conversion."""
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _int(v) -> int:
    """Safe int conversion."""
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


# ── Indices ───────────────────────────────────────────────────────────────
def _fetch_indices() -> List[Dict[str, Any]]:
    """Fetch all indices from /en/live-market/indices Drupal JSON."""
    live = _extract_drupal_settings(
        'https://www.casablanca-bourse.com/en/live-market/indices'
    )
    if not live or 'indices' not in live:
        return []

    indices = []
    for category in ['principaux', 'all']:
        for item in live['indices'].get(category, []):
            label_data = item.get('label', {})
            label = (label_data.get('en') or label_data.get('fr') or '') if isinstance(label_data, dict) else str(label_data)
            label = label.strip()

            code = item.get('code', '')
            # Skip duplicates (principaux items also appear in 'all')
            if any(i['code'] == code for i in indices):
                continue

            indices.append({
                "name":            label,
                "code":            code,
                "value":           round(_float(item.get('value', 0)), 4),
                "previous_close":  _float(item.get('previous_close', 0)),
                "variation_pct":   round(_float(item.get('change_pct', 0)), 4),
                "ytd_pct":         round(_float(item.get('change_ytd', 0)), 4),
                "high":            _float(item.get('high', 0)),
                "low":             _float(item.get('low', 0)),
            })

    return indices


# ── Overview (HTML cards: volumes, movers, transactions) ──────────────────
def _fetch_overview() -> Dict[str, Any]:
    """Fetch overview data (volumes, movers) from /en/live-market/overview HTML."""
    import re
    data = {"volumes": {}, "gainers": [], "losers": [], "transactions": []}

    try:
        r = requests.get(
            'https://www.casablanca-bourse.com/en/live-market/overview',
            headers=HEADERS, verify=False, timeout=12,
        )
        if r.status_code != 200:
            return data

        soup = BeautifulSoup(r.text, 'html.parser')

        # Volume cards
        for card in soup.find_all(class_='volume-card'):
            text = ' '.join(card.text.split())
            m = re.search(r'([\d\s]+)\s*MAD', text)
            if m:
                vol = _float(m.group(1).replace(' ', '').replace('\xa0', ''))
                if 'Cash' in text:
                    data["volumes"]["cash_volume_mad"] = vol
                elif 'Futures' in text:
                    data["volumes"]["futures_volume_mad"] = vol

        # Biggest movers
        for card in soup.find_all('div', class_='card'):
            header = card.find(class_='card-header')
            if header and 'Movers' in header.text:
                for pane in card.find_all('div', class_='tab-pane') or [card]:
                    matches = re.findall(
                        r'(\d+)\s+([A-Z0-9]+)\s+([A-Za-z\s]+?)\s+([\d\s]+,\d{2})\s+([+-]?\d+,\d{2})%',
                        pane.text,
                    )
                    for rank, ticker, name, price, var in matches:
                        entry = {
                            "rank": int(rank),
                            "ticker": ticker.strip(),
                            "name": name.strip(),
                            "last_price": _float(price.replace(' ', '').replace(',', '.')),
                            "variation_pct": _float(var.replace(' ', '').replace(',', '.').replace('+', '')),
                        }
                        if entry["variation_pct"] >= 0:
                            data["gainers"].append(entry)
                        else:
                            data["losers"].append(entry)
    except Exception as e:
        print(f"Overview scraping error: {e}")

    return data


# ── Public API ────────────────────────────────────────────────────────────

def get_market_live() -> List[Dict[str, Any]]:
    """
    Return all listed equities with full real-time data.
    Uses Drupal JSON from the BVC Equities page (30s cache).
    """
    now = time.time()
    if _equities_cache["data"] and (now - _equities_cache["ts"]) < CACHE_TTL:
        return _equities_cache["data"]

    stocks = _fetch_equities()
    if stocks:
        _equities_cache["data"] = stocks
        _equities_cache["ts"] = now
        return stocks

    # Return cached data if fetch failed
    if _equities_cache["data"]:
        return _equities_cache["data"]

    # Last-resort fallback
    return [
        {"ticker": "ATW", "name": "ATTIJARIWAFA BANK", "last_price": 545.0,
         "variation_pct": 0.0, "volume": 0, "sector": "Banks"},
        {"ticker": "BCP", "name": "BANQUE CENTRALE POPULAIRE", "last_price": 312.0,
         "variation_pct": 0.0, "volume": 0, "sector": "Banks"},
        {"ticker": "IAM", "name": "MAROC TELECOM", "last_price": 94.5,
         "variation_pct": 0.0, "volume": 0, "sector": "Telecommunications"},
    ]


def get_indices() -> List[Dict[str, Any]]:
    """
    Return all BVC indices with real-time data.
    Uses Drupal JSON from the BVC Indices page (30s cache).
    """
    now = time.time()
    if _indices_cache["data"] and (now - _indices_cache["ts"]) < CACHE_TTL:
        return _indices_cache["data"]

    indices = _fetch_indices()
    if indices:
        _indices_cache["data"] = indices
        _indices_cache["ts"] = now
        return indices

    if _indices_cache["data"]:
        return _indices_cache["data"]

    return [
        {"name": "MASI", "code": "MASI", "value": 0, "variation_pct": 0},
    ]


def get_market_overview() -> Dict[str, Any]:
    """Get overview data (volumes, movers, transactions) with 30s cache."""
    now = time.time()
    if _overview_cache["data"] and (now - _overview_cache["ts"]) < CACHE_TTL:
        return _overview_cache["data"]

    data = _fetch_overview()
    _overview_cache["data"] = data
    _overview_cache["ts"] = now
    return data


def get_stock_by_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Find a specific stock by ticker symbol or company name."""
    stocks = get_market_live()
    ticker_upper = ticker.upper()
    for s in stocks:
        if s["ticker"] == ticker_upper or s["name"].upper() == ticker_upper:
            return s
    return None


def get_top_gainers(limit: int = 10) -> List[Dict[str, Any]]:
    """Return top gaining stocks sorted by variation %."""
    stocks = [s for s in get_market_live() if _float(s.get("variation_pct")) > 0]
    stocks.sort(key=lambda x: _float(x["variation_pct"]), reverse=True)
    return stocks[:limit]


def get_top_losers(limit: int = 10) -> List[Dict[str, Any]]:
    """Return top losing stocks sorted by variation %."""
    stocks = [s for s in get_market_live() if _float(s.get("variation_pct")) < 0]
    stocks.sort(key=lambda x: _float(x["variation_pct"]))
    return stocks[:limit]


def get_most_active(limit: int = 10) -> List[Dict[str, Any]]:
    """Return most actively traded stocks by volume."""
    stocks = [s for s in get_market_live() if _float(s.get("volume")) > 0]
    stocks.sort(key=lambda x: _float(x.get("volume", 0)), reverse=True)
    return stocks[:limit]


def get_market_summary() -> Dict[str, Any]:
    """Return aggregate market summary stats."""
    stocks = get_market_live()
    overview = get_market_overview()

    gainers = [s for s in stocks if _float(s.get("variation_pct")) > 0]
    losers  = [s for s in stocks if _float(s.get("variation_pct")) < 0]
    stable  = [s for s in stocks if _float(s.get("variation_pct")) == 0]

    total_volume = sum(_float(s.get("volume", 0)) for s in stocks)
    total_cap = sum(_float(s.get("capitalisation", 0)) for s in stocks)

    # Prefer overview cash volume if available, else sum per-stock volumes
    cash_vol = overview.get("volumes", {}).get("cash_volume_mad", total_volume)

    return {
        "total_instruments": len(stocks),
        "gainers": len(gainers),
        "losers":  len(losers),
        "stable":  len(stable),
        "total_volume_mad": round(cash_vol, 2),
        "total_capitalisation_mad": round(total_cap, 2),
    }
