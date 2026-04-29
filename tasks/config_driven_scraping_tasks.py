"""
tasks/config_driven_scraping_tasks.py
Worker A Task: Config-driven universal web scraper.

Reads config/scrapers/sites.json and scrapes any enabled site automatically.
To add a new site: just add an entry to sites.json with enabled=true.
No Python code changes needed.

Queue: scraping
"""
import re
import time
import random
import hashlib
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from tasks.celery_app import celery_app
from tasks.document_ingestion_tasks import ingest_document

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent.parent / "config" / "scrapers" / "sites.json"
DOCS_DIR = Path("data/collected_data/documents")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _load_sites() -> list[dict]:
    """Load and return enabled sites from sites.json."""
    import json
    if not CONFIG_FILE.exists():
        logger.error("CONFIG: sites.json not found at %s", CONFIG_FILE)
        return []
    with open(CONFIG_FILE, encoding="utf-8") as f:
        sites = json.load(f)
    return [s for s in sites if s.get("enabled", True)]


def _polite_sleep():
    time.sleep(random.uniform(1.5, 3.0))


def _fetch_html(url: str) -> str | None:
    """Fetch a page with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20, verify=False)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(random.uniform(2, 4))
            else:
                logger.error("CONFIG SCRAPER: Failed to fetch %s: %s", url, e)
    return None


def _collect_pdf_links(listing_url: str, base_url: str, pdf_pattern: str) -> list[dict]:
    """
    Collect all PDF links from a listing page.
    Supports simple pagination (?page=N or &start=N).
    """
    entries = []
    seen = set()
    page = 0

    while True:
        paginated = f"{listing_url}{'&' if '?' in listing_url else '?'}page={page}" if page > 0 else listing_url
        html = _fetch_html(paginated)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        found = 0

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href:
                continue

            full_url = href if href.startswith("http") else base_url + href

            # Match by extension or pattern
            if href.lower().endswith(pdf_pattern) or pdf_pattern in href:
                if full_url not in seen:
                    seen.add(full_url)
                    title = link.get_text(strip=True) or Path(href).stem
                    entries.append({"title": title[:150], "pdf_url": full_url})
                    found += 1

        if found == 0:
            break  # No more pages

        page += 1
        _polite_sleep()

    return entries


def _download_pdf(pdf_url: str, title: str, site_name: str, output_folder: str) -> str | None:
    """Download a PDF and return its local path, or None on failure."""
    dest_dir = DOCS_DIR / output_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^\w\s-]", "", title[:50]).strip().replace(" ", "_")
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
    filename = f"{site_name}_{slug}_{url_hash}.pdf"
    dest = dest_dir / filename

    if dest.exists():
        logger.info("CONFIG SCRAPER: Already downloaded %s", filename)
        return str(dest.resolve())

    try:
        resp = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=60, verify=False)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        if "html" in ct and not pdf_url.lower().endswith(".pdf"):
            logger.warning("CONFIG SCRAPER: URL returned HTML, not PDF — skipping %s", title[:60])
            return None
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("CONFIG SCRAPER: Downloaded %s", filename)
        return str(dest.resolve())
    except Exception as e:
        logger.error("CONFIG SCRAPER: Download failed for %s: %s", title[:60], e)
        return None


@celery_app.task(name="tasks.config_driven_scraping_tasks.scrape_all_configured_sites")
def scrape_all_configured_sites():
    """
    Worker A Entry Task: Reads sites.json, scrapes all enabled sites,
    and dispatches discovered PDFs to the ingestion queue.
    Add sites to config/scrapers/sites.json (enabled=true) to include them.
    """
    sites = _load_sites()
    if not sites:
        logger.warning("CONFIG SCRAPER: No enabled sites found in sites.json")
        return "No enabled sites."

    logger.info("CONFIG SCRAPER: Starting scrape of %d enabled site(s).", len(sites))
    total_dispatched = 0

    for site in sites:
        name = site.get("name", "unknown")
        base_url = site.get("base_url", "")
        listing_urls = site.get("listing_urls", [])
        pdf_pattern = site.get("pdf_link_pattern", ".pdf")

        logger.info("CONFIG SCRAPER: Processing site '%s' (%d listing pages)", name, len(listing_urls))

        for listing_url in listing_urls:
            entries = _collect_pdf_links(listing_url, base_url, pdf_pattern)
            logger.info("CONFIG SCRAPER: Found %d PDFs on %s", len(entries), listing_url)

            for entry in entries:
                scrape_and_ingest_pdf.apply_async(
                    args=[entry["pdf_url"], entry["title"], name, site.get("output_folder", name)],
                    queue="scraping",
                )
                total_dispatched += 1

    return f"Dispatched {total_dispatched} PDF download tasks across {len(sites)} site(s)."


@celery_app.task(name="tasks.config_driven_scraping_tasks.scrape_and_ingest_pdf")
def scrape_and_ingest_pdf(pdf_url: str, title: str, site_name: str, output_folder: str = None):
    """Download a single PDF and route it to the ingestion worker."""
    if output_folder is None:
        output_folder = site_name
    local_path = _download_pdf(pdf_url, title, site_name, output_folder)
    if local_path:
        _polite_sleep()
        ingest_document.apply_async(args=[local_path], queue="ingestion")
