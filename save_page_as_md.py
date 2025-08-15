#!/usr/bin/env python3
"""
save_page_as_md.py — Save a webpage as Markdown with flexible extraction modes.

Install once:
    pip install readability-lxml markdownify beautifulsoup4 requests

Usage examples:
    # 1) Original behavior (best-effort "article")
    python save_page_as_md.py https://library.virginia.edu/data -o data/webpages/uvalibrary-data.md

    # 2) Try grabbing the entire <main> region (often includes multiple boxes)
    python save_page_as_md.py https://library.virginia.edu/data --mode main -o data/webpages/uvalibrary-data.md

    # 3) Use a custom CSS selector (multiple selectors allowed, comma-separated)
    python save_page_as_md.py https://library.virginia.edu/data --mode selector \
        --selector "main section, main .block__content" -o data/webpages/uvalibrary-data.md

    # 4) Convert the whole page body (minus scripts/styles)
    python save_page_as_md.py https://library.virginia.edu/data --mode full -o data/webpages/uvalibrary-data.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from readability import Document


# ------------------------------- Helpers ------------------------------------ #

def slugify(text: str) -> str:
    """Lowercase, replace non-alphanumerics with '-', trim ends."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch_html(url: str) -> str:
    """Fetch raw HTML with a friendly UA, raise for non-200s."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def absolutize_links(html: str, base_url: str) -> str:
    """Rewrite relative href/src -> absolute URLs so Markdown is portable."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(href=True):
        tag["href"] = urljoin(base_url, tag["href"])
    for tag in soup.find_all(src=True):
        tag["src"] = urljoin(base_url, tag["src"])
    return str(soup)


def extract_with_readability(html: str, url: str) -> Tuple[str, str]:
    """Main 'article' via Readability (what you had originally)."""
    doc = Document(html)
    title = doc.short_title() or url
    main_html = doc.summary(html_partial=True)
    return title, main_html


def extract_main_region(html: str, url: str) -> Tuple[str, str]:
    """Prefer <main> or [role=main]; fallback to <article>, then <body>."""
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("main, [role=main]")
    if not node:
        node = soup.select_one("article")
    if not node:
        node = soup.body or soup
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    return title, str(node)


def extract_with_selector(html: str, url: str, selector: str) -> Tuple[str, str]:
    """Concatenate all nodes that match a CSS selector (comma-separated ok)."""
    soup = BeautifulSoup(html, "htmlparser") if False else BeautifulSoup(html, "html.parser")
    nodes = soup.select(selector) if selector else []
    if not nodes:
        # Fallback to main region if nothing matches
        return extract_main_region(html, url)
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    combined = "".join(str(n) for n in nodes)
    return title, combined


def extract_full_page(html: str, url: str) -> Tuple[str, str]:
    """Whole <body> (drop scripts/styles). Good when you want *everything*."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Keep header/nav/footer if you truly want full; otherwise you could also:
    # for tag in soup.select("header, nav, footer"):
    #     tag.decompose()
    body = soup.body or soup
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    return title, str(body)


def to_markdown(url: str, mode: str, selector: str | None) -> Tuple[str, str]:
    """
    Convert page -> Markdown using chosen extraction mode, add YAML front matter.
    """
    html = fetch_html(url)

    if mode == "readability":
        title, content_html = extract_with_readability(html, url)
    elif mode == "main":
        title, content_html = extract_main_region(html, url)
    elif mode == "selector":
        title, content_html = extract_with_selector(html, url, selector or "")
    elif mode == "full":
        title, content_html = extract_full_page(html, url)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    content_html = absolutize_links(content_html, url)
    body_md = md(content_html, heading_style="ATX")

    meta = [
        "---",
        f"title: {json.dumps(title)}",  # JSON-quoted (valid YAML)
        f"source_url: {url}",
        f"downloaded: {datetime.now(timezone.utc).isoformat()}",
        f"extraction_mode: {mode}",
        f"selector: {json.dumps(selector) if selector else 'null'}",
        "---",
        "",
    ]
    markdown_text = "\n".join(meta) + body_md.strip() + "\n"
    return title, markdown_text


def default_outpath(url: str, outdir: str) -> str:
    """Deterministic filename from host+path."""
    parsed = urlparse(url)
    host = parsed.netloc or "site"
    path = parsed.path or "/"
    filename = slugify(host + path) + ".md"
    return os.path.join(outdir, filename)


# --------------------------------- CLI -------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Save a webpage as Markdown.")
    ap.add_argument("url", help="Page URL to download and convert")
    ap.add_argument("-o", "--out", help="Output .md file path (optional)")
    ap.add_argument(
        "-d", "--outdir",
        default="data/webpages",
        help="Output directory if --out is not provided (default: data/webpages)",
    )
    ap.add_argument(
        "--mode",
        choices=["readability", "main", "selector", "full"],
        default="full",
        help="Extraction mode (default: readability)",
    )
    ap.add_argument(
        "--selector",
        help="CSS selector used when --mode selector (e.g., 'main section, main .block')",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    title, md_text = to_markdown(args.url, args.mode, args.selector)
    outpath = args.out or default_outpath(args.url, args.outdir)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Saved: {outpath}")
    print(f"Title: {title}")
    print(f"Mode: {args.mode}")
    if args.selector:
        print(f"Selector: {args.selector}")


if __name__ == "__main__":
    main()
