#!/usr/bin/env python3
"""
Download the first N images from Google Images for a hardcoded query.

Features:
- Hardcoded query (TARGET), image count (N_IMAGES), and number of threads (NUM_THREADS)
- Saves to /data/$target-$timestamp/img_$n (n starts at 1)
- Multithreaded downloads with basic error handling
- Console progress reporting (tqdm)

Notes:
- This script scrapes Google Images HTML. It may break if Google changes markup
  or if access is rate-limited. Consider using an official API for robustness.
- Ensure you have permissions to write to /data on your system.

Dependencies: requests, beautifulsoup4, tqdm

Run:
  pip install requests beautifulsoup4 tqdm
  python download_google_images.py
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import re
import threading
import time
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import quote_plus

# --------------------------- Configuration ---------------------------
# Hardcoded query and parameters. Adjust as needed.
TARGET = "macbook"  # текстовый запрос (пример)
N_IMAGES = 1000      # сколько картинок скачать
NUM_THREADS = 32    # число потоков (m)

# Timeout settings (seconds)
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

# Pagination/collection controls (Google)
MAX_PAGES = 200            # upper bound for Google Images pages to visit
EMPTY_PAGE_LIMIT = 10      # stop after this many consecutive pages add no new URLs
DEFAULT_PAUSE = 0.3        # seconds between page fetches

# Additional engines and limits
ENABLE_GOOGLE = True
ENABLE_BING = True
ENABLE_DDG = True  # DuckDuckGo (via public JSON endpoint)
MAX_PAGES_BING = 200
MAX_PAGES_DDG = 200

# User-Agents to avoid immediate blocking (rotated per request)
USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
    ),
]
# The first agent kept for backwards compatibility
USER_AGENT = USER_AGENTS[0]

# --------------------------------------------------------------------

@dataclass
class DownloadResult:
    index: int
    url: str
    ok: bool
    path: str | None
    error: str | None


def slugify(value: str) -> str:
    # Simple slugify: lowercase, replace spaces with underscores, remove invalid chars
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]+", "", value)
    return value or "query"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_jpg_or_png_url(u: str) -> bool:
    # Accept common web image formats by extension: jpg/jpeg/png/webp/gif
    return re.search(r"\.(?:jpe?g|png|webp|gif)(?:$|[?#])", u, flags=re.IGNORECASE) is not None


def is_google_ui_or_thumb(u: str) -> bool:
    # Filter out Google UI assets and thumbnails
    u_low = u.lower()
    return (
        "google.com/images/branding" in u_low
        or "encrypted-tbn" in u_low  # Google thumbnails
        or "gstatic.com/images" in u_low
        or "/favicon" in u_low
    )


def google_image_search_urls(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    """Fetch candidate image URLs from Google Images HTML pages.

    Strategy:
      1) Query https://www.google.com/search?tbm=isch for successive pages (ijn=0,1,...)
      2) Parse "data-iurl" and "src" from the results grid
      3) Fallback: regex-scan for image-like URLs (jpg/png/webp)
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }

    urls: List[str] = []
    seen: Set[str] = set()
    max_pages = MAX_PAGES  # allow many pages to reach large N
    empty_pages = 0  # consecutive pages with no new urls
    stop_reason: str | None = None
    # diagnostics
    added_grid = 0
    added_ou = 0
    added_regex = 0
    pages_visited = 0

    q = quote_plus(query)
    # Progress bar for URL collection
    with tqdm(total=needed, desc="Collecting URLs", unit="url") as pbar:
        for ijn in range(max_pages):
            pages_visited += 1
            if len(urls) >= needed:
                stop_reason = "reached_needed"
                break
            # Rotate user-agents and try two pagination styles (ijn and start)
            ua = USER_AGENTS[ijn % len(USER_AGENTS)]
            headers_local = dict(headers)
            headers_local["User-Agent"] = ua
            url1 = f"https://www.google.com/search?q={q}&tbm=isch&hl=en&ijn={ijn}"
            url2 = f"https://www.google.com/search?q={q}&tbm=isch&hl=en&start={ijn*100}"
            try:
                resp = requests.get(url1, headers=headers_local, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if resp.status_code != 200 or ("consent" in resp.url and "consent" in resp.text.lower()):
                    resp = requests.get(url2, headers=headers_local, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                    if resp.status_code != 200:
                        # Try a short backoff and continue
                        time.sleep(pause + random.uniform(0, 0.3))
                        continue
            except Exception:
                time.sleep(pause + random.uniform(0, 0.3))
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            prev_len = len(urls)

            # 1) Preferred: take URLs from result grid only; ignore page UI images
            for tag in soup.select("div.isv-r img[data-iurl], div.isv-r img[src]"):
                candidate = tag.get("data-iurl") or tag.get("src") or tag.get("data-src")
                if not candidate:
                    continue
                if candidate.startswith("data:"):
                    continue
                if candidate.startswith("/"):
                    continue
                if not candidate.startswith("http"):
                    continue
                if is_google_ui_or_thumb(candidate):
                    continue
                if not is_jpg_or_png_url(candidate):
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    urls.append(candidate)
                    added_grid += 1
                    pbar.update(1)
                    if len(urls) >= needed:
                        break
            if len(urls) >= needed:
                break

            # 2) Parse JSON "ou" fields that often contain original image URLs
            if len(urls) < needed:
                ou_matches = re.findall(r'"ou":"(https?://[^"]+)"', resp.text)
                for raw in ou_matches:
                    u = raw
                    # unescape common \uXXXX sequences used by Google
                    u = u.replace("\\u003d", "=").replace("\\u0026", "&").replace("\\u002F", "/")
                    if is_google_ui_or_thumb(u):
                        continue
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
                        added_ou += 1
                        pbar.update(1)
                        if len(urls) >= needed:
                            break

            # 3) Fallback: regex-scan for common image extensions (jpg/jpeg/png/webp/gif)
            if len(urls) < needed:
                img_like = set(re.findall(r"https?://[^\s'\"<>]+\.(?:jpe?g|png|webp|gif)(?:\?[^\s'\"<>]*)?", resp.text, flags=re.IGNORECASE))
                for u in img_like:
                    if is_google_ui_or_thumb(u):
                        continue
                    if not is_jpg_or_png_url(u):
                        continue
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
                        added_regex += 1
                        pbar.update(1)
                        if len(urls) >= needed:
                            break

            if len(urls) == prev_len:
                empty_pages += 1
                if empty_pages >= EMPTY_PAGE_LIMIT:
                    stop_reason = "empty_page_limit"
                    break
            else:
                empty_pages = 0
            time.sleep(pause)

    # Determine stop reason if not set
    if stop_reason is None:
        if len(urls) >= needed:
            stop_reason = "reached_needed"
        elif pages_visited >= MAX_PAGES:
            stop_reason = "max_pages_reached"
        else:
            stop_reason = "loop_end"

    # Diagnostics output
    print(f"URL collection breakdown: pages visited={pages_visited}, from grid={added_grid}, from json-ou={added_ou}, from regex={added_regex}, total unique={len(urls)}")
    print(f"URL collection stop reason: {stop_reason} (MAX_PAGES={MAX_PAGES}, EMPTY_PAGE_LIMIT={EMPTY_PAGE_LIMIT})")
    return urls[:needed]


def ext_from_response(url: str, resp: requests.Response) -> str:
    # Try from Content-Type, then from URL
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "image/jpeg" in ct or "/jpg" in ct:
        return ".jpg"
    if "image/png" in ct:
        return ".png"
    if "image/webp" in ct:
        return ".webp"
    if "image/gif" in ct:
        return ".gif"

    m = re.search(r"\.(jpe?g|png|webp|gif)(?:\?|$)", url, flags=re.IGNORECASE)
    if m:
        return "." + m.group(1).lower()
    return ".jpg"


# --------------------------- Additional collectors ---------------------------

def collect_from_bing(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    if needed <= 0:
        return []
    urls: List[str] = []
    seen: Set[str] = set()
    pages_visited = 0
    empty_pages = 0
    added_json = 0
    q = quote_plus(query)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bing.com/",
    }
    with tqdm(total=needed, desc="Collecting URLs (Bing)", unit="url") as pbar:
        for page in range(MAX_PAGES_BING):
            if len(urls) >= needed:
                break
            pages_visited += 1
            first = page * 50 + 1
            url = f"https://www.bing.com/images/search?q={q}&first={first}&count=50&form=HDRSC2&adlt=off"
            try:
                resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if resp.status_code != 200:
                    time.sleep(pause + random.uniform(0, 0.3))
                    continue
            except Exception:
                time.sleep(pause + random.uniform(0, 0.3))
                continue
            prev_len = len(urls)
            # Extract original URLs from embedded JSON in attribute m ("murl")
            for m in re.findall(r'"murl":"(https?://[^"\\]+)"', resp.text):
                u = m.replace("\\/", "/").replace("\\u0026", "&")
                low = u.lower()
                if "mm.bing.net" in low or "/th?id=" in low:
                    continue
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
                    added_json += 1
                    pbar.update(1)
                    if len(urls) >= needed:
                        break
            if len(urls) == prev_len:
                empty_pages += 1
                if empty_pages >= EMPTY_PAGE_LIMIT:
                    break
            else:
                empty_pages = 0
            time.sleep(pause + random.uniform(0, 0.3))
    print(f"Bing breakdown: pages visited={pages_visited}, from json={added_json}, total unique={len(urls)}")
    return urls[:needed]


def collect_from_ddg(query: str, needed: int, pause: float = DEFAULT_PAUSE) -> List[str]:
    if needed <= 0:
        return []
    urls: List[str] = []
    seen: Set[str] = set()
    q = quote_plus(query)
    headers = {"User-Agent": USER_AGENT, "Referer": "https://duckduckgo.com/"}
    # Obtain vqd token
    try:
        init = requests.get(f"https://duckduckgo.com/?q={q}&iax=images&ia=images", headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        m = re.search(r"vqd='([\w-]+)'", init.text)
        if not m:
            return []
        vqd = m.group(1)
    except Exception:
        return []
    # Paginate via i.js
    s = 0
    pages_visited = 0
    empty_pages = 0
    with tqdm(total=needed, desc="Collecting URLs (DDG)", unit="url") as pbar:
        while len(urls) < needed and pages_visited < MAX_PAGES_DDG:
            pages_visited += 1
            api = (
                f"https://duckduckgo.com/i.js?l=en-us&o=json&q={q}&vqd={vqd}&f=,&p=1&s={s}"
            )
            try:
                r = requests.get(api, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break
            prev_len = len(urls)
            for item in data.get("results", []):
                u = item.get("image") or item.get("thumbnail")
                if not u:
                    continue
                if is_google_ui_or_thumb(u):
                    continue
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
                    pbar.update(1)
                    if len(urls) >= needed:
                        break
            if len(urls) == prev_len:
                empty_pages += 1
                if empty_pages >= EMPTY_PAGE_LIMIT:
                    break
            else:
                empty_pages = 0
            s += 100
            time.sleep(pause + random.uniform(0, 0.2))
    print(f"DDG breakdown: pages visited={pages_visited}, total unique={len(urls)}")
    return urls[:needed]


def collect_image_urls(query: str, needed: int) -> List[str]:
    all_urls: List[str] = []
    seen: Set[str] = set()
    # Google first
    if ENABLE_GOOGLE and needed > 0:
        g_urls = google_image_search_urls(query, needed)
        for u in g_urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
    remaining = needed - len(all_urls)
    # Bing as secondary
    if ENABLE_BING and remaining > 0:
        b_urls = collect_from_bing(query, remaining)
        for u in b_urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
    remaining = needed - len(all_urls)
    # DuckDuckGo as tertiary
    if ENABLE_DDG and remaining > 0:
        d_urls = collect_from_ddg(query, remaining)
        for u in d_urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
    print(f"Collection summary: Google={len(all_urls)}, total unique={len(all_urls)} (requested={needed})")
    return all_urls[:needed]

# --------------------------- Downloader ---------------------------

def download_one(index: int, url: str, out_dir: Path, session: requests.Session, lock: threading.Lock) -> DownloadResult:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.google.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        r = session.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True)
        if r.status_code != 200:
            return DownloadResult(index, url, False, None, f"HTTP {r.status_code}")
        # Verify content roughly by content-type or URL extension
        ct = (r.headers.get("Content-Type") or "").lower()
        is_jpeg = ("image/jpeg" in ct) or re.search(r"\.jpe?g(?:$|[?#])", url, flags=re.IGNORECASE)
        is_png  = ("image/png" in ct)  or re.search(r"\.png(?:$|[?#])", url, flags=re.IGNORECASE)
        is_webp = ("image/webp" in ct) or re.search(r"\.webp(?:$|[?#])", url, flags=re.IGNORECASE)
        is_gif  = ("image/gif" in ct)  or re.search(r"\.gif(?:$|[?#])", url, flags=re.IGNORECASE)
        if not (is_jpeg or is_png or is_webp or is_gif):
            return DownloadResult(index, url, False, None, "unsupported content-type")
        ext = ext_from_response(url, r)
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            ext = ".jpg" if is_jpeg else ".png" if is_png else ".webp" if is_webp else ".gif"
        filename = f"img_{index}{ext}"
        out_path = out_dir / filename
        # Write to disk
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return DownloadResult(index, url, True, str(out_path), None)
    except Exception as e:
        return DownloadResult(index, url, False, None, str(e))


def main():
    target = TARGET
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"data/{target}-{N_IMAGES}-{timestamp}")
    ensure_dir(out_dir)

    print(f"Target: {target}")
    print(f"Saving to: {out_dir}")
    print(f"Images requested: {N_IMAGES}; Threads: {NUM_THREADS}")

    print("Collecting image URLs ...")
    urls = collect_image_urls(target, N_IMAGES)
    print(f"Collected {len(urls)} candidate URLs")

    # Prepare session and download pool
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: List[DownloadResult] = []
    lock = threading.Lock()

    with tqdm(total=len(urls), desc="Downloading", unit="img") as pbar:
        with cf.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(download_one, i + 1, url, out_dir, session, lock)
                for i, url in enumerate(urls)
            ]
            for future in cf.as_completed(futures):
                res: DownloadResult = future.result()
                results.append(res)
                if res.ok:
                    pbar.update(1)
                else:
                    # Consider failed downloads still move the bar to reflect attempts
                    pbar.update(1)
                    pbar.set_postfix_str("failures present")

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print(f"Done. Success: {ok_count}, Failed: {fail_count}")
    if fail_count:
        print("Sample failures:")
        for r in results:
            if not r.ok:
                print(f"  #{r.index}: {r.url} -> {r.error}")
                # limit printed failures
                fail_count -= 1
                if fail_count <= 0:
                    break


if __name__ == "__main__":
    main()
