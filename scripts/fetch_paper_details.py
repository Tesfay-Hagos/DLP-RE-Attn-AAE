#!/usr/bin/env python3
"""
Fetch full paper details by DOI or arXiv ID.

Sources tried in order:
  1. Semantic Scholar API  — title, authors, year, abstract, venue, citations
  2. Crossref API          — fallback for DOI-only papers (no abstract)
  3. arXiv API             — fallback for arXiv-only papers

Papers that fail all sources are written to manual_lookup.txt with their URL.

Usage:
  python fetch_paper_details.py                  # uses built-in papers list
  python fetch_paper_details.py --input my.json  # use custom input JSON
  python fetch_paper_details.py --out results.json --manual manual.txt
"""

import json
import time
import requests
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# ── Papers to fetch ────────────────────────────────────────────────────────────
# Format: {"id": "...", "doi": "...", "arxiv": "...", "role": "...", "note": "..."}
# Provide at least one of doi or arxiv.
PAPERS = [
    # ── Core dataset ──────────────────────────────────────────────────────────
    {
        "id": "ham10000",
        "doi": "10.1038/sdata.2018.161",
        "role": "Dataset — HAM10000 10K dermoscopy images, 7 classes, ISIC 2018"
    },
    {
        "id": "isic2018",
        "arxiv": "1902.03368",
        "role": "ISIC 2018 Task3 challenge — defines balanced accuracy metric"
    },
    # ── Core SSL / semi-supervised methods ────────────────────────────────────
    {
        "id": "simclr",
        "arxiv": "2002.05709",
        "role": "SimCLR — NT-Xent contrastive pretraining (our Stage 1)"
    },
    {
        "id": "fixmatch",
        "arxiv": "2001.07685",
        "role": "FixMatch — hard pseudo-label SSL (our Stage 2)"
    },
    {
        "id": "mean_teacher",
        "arxiv": "1703.01780",
        "role": "Mean Teacher — EMA soft consistency (predecessor / baseline)"
    },
    # ── Skin lesion SSL competitors ───────────────────────────────────────────
    {
        "id": "abcl",
        "doi": "10.1016/j.compbiomed.2022.105676",
        "role": "ABCL — asymmetric class-aware SSL for skin lesion (competitor)"
    },
    {
        "id": "csda",
        "arxiv": "2307.15987",
        "role": "CSDA — class-aware semi-supervised DA for skin lesion (competitor)"
    },
    {
        "id": "ssl_skin_eccv",
        "doi": "10.1007/978-3-031-25069-9_11",
        "arxiv": "2106.09229",
        "role": "ECCV 2022 — evaluation of SSL pretraining for skin lesion analysis"
    },
    {
        "id": "ncpl",
        "doi": "10.1007/978-3-031-47425-5_22",
        "role": "NCPL MICCAI 2023 — noisy consistent pseudo-labeling for skin lesion"
    },
    # ── Architecture / infrastructure ─────────────────────────────────────────
    {
        "id": "efficientnet",
        "arxiv": "1905.11946",
        "role": "EfficientNet — backbone architecture (B3 used throughout)"
    },
    {
        "id": "optuna",
        "arxiv": "1907.10902",
        "role": "Optuna — Bayesian HP search (TPESampler + MedianPruner, 30 trials)"
    },
    {
        "id": "imagenet",
        "doi": "10.1007/s11263-015-0816-y",
        "role": "ImageNet Large Scale Visual Recognition Challenge — pretraining source"
    },
    # ── Semi-supervised general context ───────────────────────────────────────
    {
        "id": "flexmatch",
        "arxiv": "2110.08263",
        "role": "FlexMatch — curriculum pseudo-labeling, per-class adaptive threshold (NeurIPS 2021)"
    },
]

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper"
CROSSREF_BASE         = "https://api.crossref.org/works"
ARXIV_BASE            = "http://export.arxiv.org/api/query"

FIELDS = "title,authors,year,abstract,venue,externalIds,citationCount,publicationVenue"


def _get(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "paper-fetcher/1.0 (research)"})
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print("    ⏳ Rate limited — sleeping 12s...")
            time.sleep(12)
            r = requests.get(url, params=params, timeout=timeout)
            return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"    ⚠️  Request error: {e}")
    return None


def fetch_semantic_scholar(doi: str = None, arxiv: str = None) -> Optional[dict]:
    """Try DOI then arXiv ID on Semantic Scholar."""
    identifiers = []
    if doi:
        identifiers.append(f"DOI:{doi}")
    if arxiv:
        identifiers.append(f"ARXIV:{arxiv}")

    for identifier in identifiers:
        url  = f"{SEMANTIC_SCHOLAR_BASE}/{identifier}"
        data = _get(url, params={"fields": FIELDS})
        if data and "title" in data:
            authors = [a.get("name", "") for a in data.get("authors", [])]
            venue   = (data.get("publicationVenue") or {}).get("name") or data.get("venue") or ""
            eids    = data.get("externalIds") or {}
            return {
                "title":    data.get("title", ""),
                "authors":  authors,
                "year":     data.get("year"),
                "abstract": data.get("abstract", ""),
                "venue":    venue,
                "citations":data.get("citationCount", 0),
                "doi":      eids.get("DOI") or doi,
                "arxiv":    eids.get("ArXiv") or arxiv,
                "source":   "SemanticScholar",
                "s2_url":   f"https://api.semanticscholar.org/graph/v1/paper/{identifier}",
            }
        time.sleep(1)
    return None


def fetch_crossref(doi: str) -> Optional[dict]:
    """Fetch metadata from Crossref (title, authors, year, venue — no abstract)."""
    data = _get(f"{CROSSREF_BASE}/{doi}")
    if not data:
        return None
    msg = data.get("message", {})
    if not msg.get("title"):
        return None

    authors = []
    for a in msg.get("author", []):
        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if name:
            authors.append(name)

    year = None
    for date_field in ["published-print", "published-online", "created"]:
        dp = msg.get(date_field, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break

    venue = (msg.get("container-title") or [""])[0]
    return {
        "title":    msg["title"][0] if isinstance(msg["title"], list) else msg["title"],
        "authors":  authors,
        "year":     year,
        "abstract": "",  # Crossref rarely has abstracts
        "venue":    venue,
        "citations":0,
        "doi":      doi,
        "arxiv":    None,
        "source":   "Crossref",
        "s2_url":   f"https://doi.org/{doi}",
    }


def fetch_arxiv(arxiv_id: str) -> Optional[dict]:
    """Fetch metadata from arXiv API."""
    try:
        r = requests.get(ARXIV_BASE,
                         params={"id_list": arxiv_id, "max_results": 1},
                         timeout=15)
        if r.status_code != 200:
            return None
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title   = (entry.findtext("atom:title", "", ns) or "").replace("\n", " ").strip()
        summary = (entry.findtext("atom:summary", "", ns) or "").replace("\n", " ").strip()
        year    = None
        pub_raw = entry.findtext("atom:published", "", ns)
        if pub_raw:
            year = int(pub_raw[:4])
        authors = [a.findtext("atom:name", "", ns)
                   for a in entry.findall("atom:author", ns)]
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        doi    = doi_el.text if doi_el is not None else None

        return {
            "title":    title,
            "authors":  authors,
            "year":     year,
            "abstract": summary,
            "venue":    "arXiv",
            "citations":0,
            "doi":      doi,
            "arxiv":    arxiv_id,
            "source":   "arXiv",
            "s2_url":   f"https://arxiv.org/abs/{arxiv_id}",
        }
    except Exception as e:
        print(f"    ⚠️  arXiv fetch error: {e}")
        return None


def fetch_paper(paper_def: dict) -> Optional[dict]:
    """Try all sources for a paper definition. Returns enriched dict or None."""
    doi   = paper_def.get("doi")
    arxiv = paper_def.get("arxiv")

    print(f"\n  [{paper_def['id']}]  doi={doi}  arxiv={arxiv}")

    # 1. Semantic Scholar (best — has abstract + citations)
    result = fetch_semantic_scholar(doi=doi, arxiv=arxiv)
    if result:
        print(f"    ✅ SemanticScholar — {result['title'][:70]}")
        result.update({"id": paper_def["id"], "role": paper_def.get("role", "")})
        return result
    time.sleep(1.5)

    # 2. Crossref (DOI only)
    if doi:
        result = fetch_crossref(doi)
        if result:
            print(f"    ✅ Crossref (no abstract) — {result['title'][:70]}")
            result.update({"id": paper_def["id"], "role": paper_def.get("role", "")})
            return result
        time.sleep(1)

    # 3. arXiv
    if arxiv:
        result = fetch_arxiv(arxiv)
        if result:
            print(f"    ✅ arXiv — {result['title'][:70]}")
            result.update({"id": paper_def["id"], "role": paper_def.get("role", "")})
            return result

    print(f"    ❌ All sources failed")
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch full paper details by DOI/arXiv ID")
    parser.add_argument("--input",  default=None,
                        help="JSON file with paper definitions (default: built-in list)")
    parser.add_argument("--out",    default="skin cancer/fetched_papers.json",
                        help="Output JSON file")
    parser.add_argument("--manual", default="skin cancer/manual_lookup.txt",
                        help="Output file for papers that need manual lookup")
    args = parser.parse_args()

    papers_to_fetch = PAPERS
    if args.input:
        with open(args.input) as f:
            papers_to_fetch = json.load(f)

    print("=" * 70)
    print("  PAPER DETAIL FETCHER")
    print(f"  Fetching {len(papers_to_fetch)} papers")
    print("=" * 70)

    fetched  = []
    failed   = []

    for p in papers_to_fetch:
        result = fetch_paper(p)
        if result:
            fetched.append(result)
        else:
            failed.append(p)
        time.sleep(1.5)  # be polite to APIs

    # Save fetched results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fetched, f, indent=2)
    print(f"\n✅ Saved {len(fetched)} papers → {out_path}")

    # Save manual lookup list
    manual_path = Path(args.manual)
    with open(manual_path, "w") as f:
        f.write("PAPERS REQUIRING MANUAL LOOKUP\n")
        f.write("=" * 60 + "\n\n")
        if failed:
            for p in failed:
                f.write(f"ID   : {p['id']}\n")
                f.write(f"Role : {p.get('role', '')}\n")
                if p.get("doi"):
                    f.write(f"URL  : https://doi.org/{p['doi']}\n")
                if p.get("arxiv"):
                    f.write(f"URL  : https://arxiv.org/abs/{p['arxiv']}\n")
                f.write("Copy: title, authors, year, venue, abstract\n")
                f.write("-" * 60 + "\n\n")
        else:
            f.write("All papers fetched successfully — no manual lookup needed.\n")

    print(f"{'✅' if not failed else '⚠️ '} Manual lookup file → {manual_path}")
    if failed:
        print(f"   {len(failed)} papers need manual lookup: {[p['id'] for p in failed]}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'ID':<30} {'Source':<18} {'Year'} {'Citations'}")
    print("-" * 70)
    for r in fetched:
        print(f"  {r['id']:<30} {r['source']:<18} {r.get('year','?')!s:<6} {r.get('citations',0)}")
    if failed:
        for p in failed:
            print(f"  {p['id']:<30} {'❌ FAILED':<18}")
    print("=" * 70)


if __name__ == "__main__":
    main()
