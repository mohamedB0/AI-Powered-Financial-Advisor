import requests
import json
import re
import time
import warnings
from typing import List, Dict, Any, Optional
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
}

_build_id_cache: Dict[str, Any] = {"id": None, "ts": 0}

def _get_build_id() -> Optional[str]:
    now = time.time()
    if _build_id_cache["id"] and (now - _build_id_cache["ts"]) < 3600:
        return _build_id_cache["id"]
    try:
        r = requests.get("https://www.casablanca-bourse.com/fr",
                         headers=HEADERS, verify=False, timeout=15)
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                          r.text)
        if match:
            build_id = json.loads(match.group(1)).get("buildId")
            if build_id:
                _build_id_cache["id"] = build_id
                _build_id_cache["ts"] = now
                return build_id
    except Exception as e:
        print(f"BuildId error: {e}")
    return _build_id_cache.get("id")

def get_market_live() -> List[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/ticker",
            params={"marche": 59, "class[]": 50},
            headers=HEADERS, verify=False, timeout=15
        )
        if r.status_code == 200:
            stocks = r.json()["data"]["values"]
            result = []
            for s in stocks:
                result.append({
                    "ticker":         s.get("ticker", ""),
                    "name":           s.get("label", ""),
                    "sector":         s.get("sous_secteur", ""),
                    "last_price":     s.get("field_cours_courant"),
                    "ref_price":      s.get("field_static_reference_price"),
                    "open":           s.get("field_opening_price"),
                    "high":           s.get("field_high_price"),
                    "low":            s.get("field_low_price"),
                    "variation_pct":  s.get("field_var_veille"),
                    "volume":         s.get("field_cumul_volume_echange"),
                    "qty_traded":     s.get("field_cumul_titres_echanges"),
                    "nb_trades":      s.get("field_total_trades"),
                    "capitalisation": s.get("field_capitalisation"),
                    "status":         s.get("field_etat_cot_val"),
                    "bid_price":      s.get("field_best_bid_price"),
                    "ask_price":      s.get("field_best_ask_price"),
                })
            return result
    except Exception as e:
        print(f"Market live error: {e}")
    return []

def get_stock_by_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    stocks = get_market_live()
    ticker_upper = ticker.upper()
    for s in stocks:
        if s["ticker"] == ticker_upper or s["name"].upper() == ticker_upper:
            return s
    return None

def get_indices() -> List[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://www.casablanca-bourse.com/api/proxy/fr/api/bourse/dashboard/grouped_index_watch",
            headers=HEADERS, verify=False, timeout=15
        )
        if r.status_code == 200:
            raw = r.json().get("data", [])
            result = []
            for category in raw:
                cat_name = category.get("title", "")
                for item in category.get("items", []):
                    index_url = item.get("index_url", "")
                    code = index_url.split("/")[-1] if index_url else ""
                    result.append({
                        "category":       cat_name,
                        "name":           item.get("index", ""),
                        "code":           code,
                        "value":          item.get("field_index_value"),
                        "previous":       item.get("veille"),
                        "variation_pct":  item.get("field_var_veille"),
                        "variation_ytd":  item.get("field_var_year"),
                        "high":           item.get("field_index_high_value"),
                        "low":            item.get("field_index_low_value"),
                        "capitalisation": item.get("field_market_capitalisation"),
                    })
            return result
    except Exception as e:
        print(f"Indices error: {e}")
    return []

def get_top_gainers(limit: int = 10) -> List[Dict[str, Any]]:
    stocks = [s for s in get_market_live() if s["variation_pct"] is not None and float(s["variation_pct"]) > 0]
    stocks.sort(key=lambda x: float(x["variation_pct"]), reverse=True)
    return stocks[:limit]

def get_top_losers(limit: int = 10) -> List[Dict[str, Any]]:
    stocks = [s for s in get_market_live() if s["variation_pct"] is not None and float(s["variation_pct"]) < 0]
    stocks.sort(key=lambda x: float(x["variation_pct"]))
    return stocks[:limit]

def get_most_active(limit: int = 10) -> List[Dict[str, Any]]:
    stocks = [s for s in get_market_live() if s["volume"] is not None]
    stocks.sort(key=lambda x: float(x["volume"] or 0), reverse=True)
    return stocks[:limit]

def get_market_summary() -> Dict[str, Any]:
    stocks = get_market_live()
    gainers = [s for s in stocks if s["variation_pct"] is not None and float(s["variation_pct"]) > 0]
    losers  = [s for s in stocks if s["variation_pct"] is not None and float(s["variation_pct"]) < 0]
    # Stable: explicitly 0 variation; no_data: variation_pct is None (not yet traded)
    stable  = [s for s in stocks if s["variation_pct"] is not None and float(s["variation_pct"]) == 0]
    no_data = [s for s in stocks if s["variation_pct"] is None]
    total_vol = sum(float(s["volume"] or 0) for s in stocks)
    total_cap = sum(float(s["capitalisation"] or 0) for s in stocks)
    return {
        "total_instruments": len(stocks),
        "gainers": len(gainers),
        "losers":  len(losers),
        "stable":  len(stable) + len(no_data),   # UI shows both as "="
        "total_volume_mad": round(total_vol, 2),
        "total_capitalisation_mad": round(total_cap, 2),
    }