"""
ingestion/loaders/bvc_loader.py
Fetches structured market data from Bourse de Casablanca (BVC).
Stores OHLCV data in PostgreSQL via StockPrice model.
"""

import logging
import re
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    logger.warning("pandas not installed, BVCLoader will be limited")


BVC_BASE_URL = "https://www.casablanca-bourse.com"
BVC_COTATION_URL = f"{BVC_BASE_URL}/fr/cotation/actions"
BVC_INDICES_URL = f"{BVC_BASE_URL}/fr/cotation/indices"

BVC_TICKER_MAP = {
    "BCP": "Banque Populaire",
    "MTD": "Mutuelle des Telecom",
    "IAM": "Maroc Telecom",
    "attijari": "Attijariwafa Bank",
    "CIH": "CIH Bank",
    "STM": "STMicroelectronics",
    "ODS": "Oulmaz",
    "AFMA": "Afma",
    "ALM": "Aluminum du Maroc",
    "Aradei": "Aradei Capital",
    "Awqaf": "Awqaf Salaf",
    "Bank of Africa": "Bank of Africa",
    "Beni Amir": "Beni Amir",
    "Caras": "Caras",
    "CDG": "CDG",
    "Colombie": "Colombie",
    "CTM": "CTM",
    "Dari Cous": "Dari Cous",
    "Deli Berliner": "Deli Berliner",
    "Delta": "Delta Holding",
    "Diac": "Diac",
    "Diff": "Diff",
    "Djoum": "Djoum",
    "Dolidol": "Dolidol",
    "FENIX": "Fenix",
    "HPS": "HPS",
    "IBmaroc": "IB Maroc",
    "Immoben": "Immoben",
    "Involys": "Involys",
    "Itissal": "Itissal Al-Maghrib",
    "Jetmar": "Jetmar",
    "K": "K",
    "KRAF": "KRAF",
    "Kromberg": "Kromberg",
    "Lafarge": "LafargeHolcim",
    "Lek": "Lek",
    "Lesieur": "Lesieur",
    "M2": "M2",
    "MAB": "MAB",
    "Managem": "Managem",
    "Maroc": "Maroc",
    "Mediac": "Mediac",
    "Microdata": "Microdata",
    "Mocr": "Mocr",
    "NEX": "NEX",
    "Oulmaz": "Oulmaz",
    "Phoenix": "Phoenix",
    "Rism": "Rism",
    "S2M": "S2M",
    "SCR": "SCR",
    "SMT": "SMT",
    "Sonasid": "Sonad",
    "Sonasid": "Sonasid",
    "SRM": "SRM",
    "Stokvis": "Stokvis",
    "Tagner": "Tagner",
    "Timar": "Timar",
    "Total": "Total Maroc",
    "Travel": "Travel",
    "T杂": "T杂",
    "UPS": "UPS",
    "Zellidja": "Zellidja",
}


class BVCLoader:
    """
    Fetches market data from Bourse de Casablanca.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        })

    def fetch_cotation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Fetch daily stock cotation (prices) from BVC.
        Returns (text_records, price_df) for vector store and SQL.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required: pip install pandas")

        logger.info("BVCLoader: fetching cotation from %s", BVC_COTATION_URL)

        try:
            response = self.session.get(BVC_COTATION_URL, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as e:
            logger.error("BVCLoader: failed to fetch cotation page: %s", e)
            return [], pd.DataFrame()

        dfs = pd.read_html(StringIO(html), encoding="utf-8")
        if not dfs:
            logger.warning("BVCLoader: no HTML tables found on cotation page")
            return [], pd.DataFrame()

        df = dfs[0]
        logger.info("BVCLoader: extracted table with %d rows", len(df))

        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")

        price_records = []
        text_records = []

        for _, row in df.iterrows():
            try:
                ticker = str(row.get("Symbole", row.get("Sym", ""))).strip()
                if not ticker or ticker == "nan":
                    continue

                close_val = row.get("Cours Clôture", row.get("Clôture", row.get("Last", None)))
                open_val = row.get("Premier", row.get("Open", row.get("Ouverture", None)))
                high_val = row.get("Plus Haut", row.get("High", row.get(" Haut", None)))
                low_val = row.get("Plus Bas", row.get("Low", row.get("Bas", None)))
                volume_val = row.get("Volume", row.get("Vol", 0))

                if close_val is None or str(close_val) == "nan":
                    continue

                price_date = datetime.now().date()
                if "Date" in row:
                    try:
                        price_date = pd.to_datetime(row["Date"]).date()
                    except Exception:
                        pass

                record = {
                    "ticker": ticker,
                    "date": price_date,
                    "open": self._parse_float(open_val),
                    "high": self._parse_float(high_val),
                    "low": self._parse_float(low_val),
                    "close": self._parse_float(close_val),
                    "volume": int(self._parse_float(volume_val) or 0),
                }
                price_records.append(record)

                company_name = BVC_TICKER_MAP.get(ticker, ticker)
                text_parts = [
                    f"Stock: {company_name} ({ticker})",
                    f"Close: {close_val}",
                    f"Open: {open_val}",
                    f"High: {high_val}",
                    f"Low: {low_val}",
                    f"Volume: {volume_val}",
                    f"Date: {price_date}",
                    f"Source: Bourse de Casablanca",
                ]
                text_records.append(("\n".join(text_parts), {"source": f"bvc:{ticker}", "doc_type": "structured"}))
            except Exception as e:
                logger.debug("BVCLoader: skipped row due to: %s", e)
                continue

        if price_records:
            df_prices = pd.DataFrame(price_records)
            self._store_prices(df_prices)
        else:
            df_prices = pd.DataFrame()

        return text_records, df_prices

    def fetch_indices(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Fetch BVC index values.
        Returns (text_records, index_df) for vector store and SQL.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required: pip install pandas")

        logger.info("BVCLoader: fetching indices from %s", BVC_INDICES_URL)

        try:
            response = self.session.get(BVC_INDICES_URL, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as e:
            logger.error("BVCLoader: failed to fetch indices page: %s", e)
            return [], pd.DataFrame()

        dfs = pd.read_html(StringIO(html), encoding="utf-8")
        if not dfs:
            logger.warning("BVCLoader: no HTML tables found on indices page")
            return [], pd.DataFrame()

        df = dfs[0]
        logger.info("BVCLoader: extracted indices table with %d rows", len(df))

        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")

        index_records = []
        text_records = []

        for _, row in df.iterrows():
            try:
                index_name = str(row.get("Indice", row.get("Index", ""))).strip()
                if not index_name or index_name == "nan":
                    continue

                value = row.get("Valeur", row.get("Value", row.get("Niveau", None)))
                if value is None or str(value) == "nan":
                    continue

                variation = row.get("Variation", row.get("%", 0))
                date_val = datetime.now().date()

                record = {
                    "country": "Morocco",
                    "indicator_name": f"BVC Index: {index_name}",
                    "value": self._parse_float(value),
                    "unit": "points",
                    "date": date_val,
                    "source": "Bourse de Casablanca",
                }
                index_records.append(record)

                text_parts = [
                    f"Index: {index_name}",
                    f"Value: {value}",
                    f"Variation: {variation}%",
                    f"Date: {date_val}",
                    f"Source: Bourse de Casablanca",
                ]
                text_records.append(("\n".join(text_parts), {"source": f"bvc:index:{index_name}", "doc_type": "structured"}))
            except Exception as e:
                logger.debug("BVCLoader: skipped index row due to: %s", e)
                continue

        if index_records:
            df_indices = pd.DataFrame(index_records)
            self._store_macro(df_indices)
        else:
            df_indices = pd.DataFrame()

        return text_records, df_indices

    def fetch_all(self) -> Dict[str, Any]:
        """
        Fetch all BVC data: cotation and indices.
        Returns summary dict with status.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "source": "Bourse de Casablanca",
            "cotation": {"status": "pending", "records": 0},
            "indices": {"status": "pending", "records": 0},
        }

        try:
            text_records, price_df = self.fetch_cotation()
            results["cotation"] = {
                "status": "success",
                "records": len(price_df) if not price_df.empty else 0,
                "tickers": len(price_df["ticker"].unique()) if not price_df.empty else 0,
            }
        except Exception as e:
            results["cotation"] = {"status": "error", "error": str(e)}

        try:
            text_records, index_df = self.fetch_indices()
            results["indices"] = {
                "status": "success",
                "records": len(index_df) if not index_df.empty else 0,
            }
        except Exception as e:
            results["indices"] = {"status": "error", "error": str(e)}

        return results

    def _parse_float(self, val: Any) -> Optional[float]:
        """Parse a value to float, handling various formats."""
        if val is None or str(val).strip() == "" or str(val).lower() == "nan":
            return None
        try:
            val_str = str(val).replace(",", ".").replace(" ", "").replace("\xa0", "")
            return float(val_str)
        except (ValueError, AttributeError):
            return None

    def _store_prices(self, df: pd.DataFrame) -> int:
        """Store price records in PostgreSQL via RelationalDB."""
        if df.empty:
            return 0
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()
            records = df.to_dict(orient="records")
            count = db.insert_prices(records)
            logger.info("BVCLoader: stored %d price records in PostgreSQL", count)
            return count
        except Exception as e:
            logger.error("BVCLoader: failed to store prices: %s", e)
            return 0

    def _store_macro(self, df: pd.DataFrame) -> int:
        """Store index records as macro indicators in PostgreSQL."""
        if df.empty:
            return 0
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()
            records = df.to_dict(orient="records")
            count = db.insert_macro(records)
            logger.info("BVCLoader: stored %d macro/index records in PostgreSQL", count)
            return count
        except Exception as e:
            logger.error("BVCLoader: failed to store macro data: %s", e)
            return 0
